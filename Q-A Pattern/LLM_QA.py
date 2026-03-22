
# python LLM_QA.py --input DesignPairSession_03-30-2023.json

import json, os, re
from pathlib import Path

import groq
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GROQ_API_KEY in your env or .env")

MODEL = "openai/gpt-oss-20b"
LONG_SEG_SEC = 20.0

PASS1_SYSTEM = """You are a meeting analyst.
Input is a transcript chunk with segment IDs, timestamps, and speakers.

Task: Identify ALL utterances that function as questions (information-seeking / confirmation-seeking / decision-seeking),
even if not phrased with '?'.

Include proposal-style questions too (these ARE questions):
- "Can we...?"
- "Should we...?"
- "Could we...?"
- "Do we want to...?"
- "Would it be better if...?"

Avoid false positives:
- Do NOT create questions from agenda statements or explanations.
- Only mark a segment as a question if the speaker is actually asking others for info/confirmation/decision.
- If a segment contains both explanation and a question, extract ONLY the question portion into question_text_clean.

Exclude purely rhetorical questions if clearly not seeking an answer and nobody responds.

CRITICAL:
- Always return valid JSON matching the schema.
- If there are ZERO questions in this chunk, return exactly: {"questions": []}
- question_id MUST be the segment id in the form "#000123" (six digits).

Output ONLY JSON. No extra text.
"""

PASS2_SYSTEM = """You extract the answer for ONE given question from a transcript episode.

Rules:
- Answer = any utterance that addresses the question (may be partial, multi-speaker).
- Cite evidence using ONLY segment IDs from the episode.
- If no answer, answer_status="unanswered" and answer_evidence_ids=[].
- Do NOT add external facts.

CRITICAL:
- Always return valid JSON matching the schema.
- Evidence ids MUST be segment ids in the form "#000123" (six digits).
- If answer_status is "answered" or "partial", you MUST include at least one evidence id.

Output ONLY JSON. No extra text.
"""

QUESTION_TYPES = ["clarification","process","decision","knowledge","planning","status","ownership","scope","other"]

PASS1_SCHEMA = {
    "type":"object",
    "properties":{"questions":{"type":"array","items":{
        "type":"object",
        "properties":{
            "question_id":{"type":"string"},
            "question_text_clean":{"type":"string"},
            "question_type":{"type":"string","enum":QUESTION_TYPES},
            "is_rhetorical":{"type":"boolean"},
            "confidence":{"type":"number"},
        },
        "required":["question_id","question_text_clean","question_type","is_rhetorical","confidence"],
        "additionalProperties":False
    }}},
    "required":["questions"],
    "additionalProperties":False
}

PASS2_SCHEMA = {
    "type":"object",
    "properties":{
        "answer_status":{"type":"string","enum":["answered","partial","unanswered"]},
        "answer_summary":{"type":"string"},  
        "answer_evidence_ids":{"type":"array","items":{"type":"string"}},
        "confidence":{"type":"number"},
        "support_check_passed":{"type":"boolean"},
    },
    "required":["answer_status","answer_summary","answer_evidence_ids","confidence","support_check_passed"],
    "additionalProperties":False
}

def ts(sec: float) -> str:
    h = int(sec//3600); m = int((sec%3600)//60); s = sec%60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def tok_est(s: str) -> int:
    return max(1, len(s)//4)

def norm_id(x):
    s = str(x).strip()
    m = re.search(r"#\s*(\d+)", s)
    if not m:
        m = re.search(r"\b(\d+)\b", s)
    if not m:
        return None
    return f"#{int(m.group(1)):06d}"

def groq_json(client: Groq, system: str, user: str, schema: dict, name: str):
    def default():
        if name == "q_detect":
            return {"questions": []}
        return {"answer_status":"unanswered","answer_summary":"","answer_evidence_ids":[],"confidence":0.0,"support_check_passed":False}

    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0,
            response_format={"type":"json_schema","json_schema":{"name":name,"strict":True,"schema":schema}},
        )
        return json.loads(r.choices[0].message.content)
    except groq.BadRequestError:
        
        try:
            r2 = client.chat.completions.create(
                model=MODEL,
                messages=[{"role":"system","content":system},{"role":"user","content":user}],
                temperature=0,
                response_format={"type":"json_object"},
            )
            return json.loads(r2.choices[0].message.content)
        except Exception:
            return default()

def load_segments(p: Path):
    raw = json.loads(p.read_text(encoding="utf-8"))
    segs = raw if isinstance(raw, list) else (raw.get("segments") or raw.get("utterances") or raw.get("items") or raw.get("results"))
    if not isinstance(segs, list):
        raise ValueError("Unrecognized JSON structure (expected list or {segments:[...]})")
    out = []
    for s in segs:
        if not isinstance(s, dict):
            continue
        sp = str(s.get("speaker","")).strip()
        tx = str(s.get("text","")).strip()
        if not sp or not tx:
            continue
        try:
            st = float(s.get("start",0.0)); en = float(s.get("end",st))
        except:
            continue
        if en < st: en = st
        out.append({"speaker":sp,"start":st,"end":en,"text":tx})
    out.sort(key=lambda x:(x["start"],x["end"]))
    return out

def canonicalize(segs, merge_gap=1.5):
    merged = []
    for s in segs:
        if not merged:
            merged.append(dict(s)); continue
        p = merged[-1]
        if p["speaker"]==s["speaker"] and 0 <= s["start"]-p["end"] <= merge_gap:
            p["end"] = max(p["end"], s["end"])
            p["text"] = (p["text"].rstrip()+" "+s["text"].lstrip()).strip()
        else:
            merged.append(dict(s))
    canon = []
    for i,m in enumerate(merged):
        canon.append({
            "id": f"#{i:06d}",
            "speaker": m["speaker"],
            "start": float(m["start"]),
            "end": float(m["end"]),
            "text": m["text"].strip(),
        })
    return canon

def line(seg):
    return f"[{seg['id']} {ts(seg['start'])}-{ts(seg['end'])}] {seg['speaker']}: {seg['text']}"

def chunk_lines(canon, target_tokens=20000, overlap_frac=0.18):
    overlap_tokens = int(target_tokens*overlap_frac)
    chunks = []
    i, n = 0, len(canon)
    while i < n:
        start = i; t = 0; lines = []
        while i < n:
            ln = line(canon[i]); lt = tok_est(ln)
            if lines and t+lt > target_tokens: break
            lines.append(ln); t += lt; i += 1
        end = i
        chunks.append((start, end, "\n".join(lines)))
        if i >= n: break
        if overlap_tokens > 0:
            back = 0; j = end-1
            while j > start and back < overlap_tokens:
                back += tok_est(line(canon[j])); j -= 1
            i = max(0, j)
    return chunks

def build_episode(canon, id_to_idx, qid, pre_turns=2, window_sec=150, max_turns=120):
    qi = id_to_idx[qid]
    start = max(0, qi-pre_turns)
    cutoff = canon[qi]["start"] + window_sec
    ep = []
    i = start
    while i < len(canon) and len(ep) < max_turns:
        ep.append(canon[i])
        if canon[i]["start"] > cutoff: break
        i += 1
    return "\n".join(line(s) for s in ep)

def expand_evidence(canon, id_to_idx, by_id, qid, evid_ids):
    """Small, bounded expansion to reduce truncated answers."""
    if not evid_ids:
        return evid_ids

    q_idx = id_to_idx[qid]
    idxs = sorted({id_to_idx[e] for e in evid_ids if e in id_to_idx})

    filled = set(idxs)
    for a, b in zip(idxs, idxs[1:]):
        gap = b - a
        if 1 < gap <= 3:  
            sa = canon[a]["speaker"]; sb = canon[b]["speaker"]
            for k in range(a+1, b):
                if canon[k]["speaker"] in (sa, sb):
                    filled.add(k)

    last = max(filled)
    last_sp = canon[last]["speaker"]
    prev_end = canon[last]["end"]
    k = last + 1
    added = 0
    while k < len(canon) and added < 3:
        if canon[k]["speaker"] != last_sp:
            break
        if canon[k]["start"] - prev_end > 2.0:
            break
        filled.add(k)
        prev_end = canon[k]["end"]
        added += 1
        k += 1

    out_ids = [canon[i]["id"] for i in sorted(filled) if i >= q_idx or canon[i]["id"] == qid]
    seen = set()
    out = []
    for x in out_ids:
        if x not in seen and x in by_id:
            seen.add(x); out.append(x)
    return out

def is_questionish(text: str) -> bool:
    t = (text or "").strip().lower()
    if "?" in t:
        return True
    starters = ("what ","why ","how ","when ","where ","who ","which ",
                "can ","could ","should ","do ","does ","did ","would ",
                "is ","are ","am ","will ")
    if t.startswith(starters):
        return True
    if " i have a question" in (" " + t) or t.startswith("i have a question"):
        return True
    return False

def pick_reply_evidence(canon, id_to_idx, qid, asker_speaker, max_ids=3, max_after_sec=90.0):
    """If model fails, prefer nearby replies by other speakers (non-question)."""
    qi = id_to_idx[qid]
    q_end = canon[qi]["end"]
    out = []
    k = qi + 1
    while k < len(canon) and len(out) < max_ids:
        if canon[k]["start"] - q_end > max_after_sec:
            break
        if canon[k]["speaker"] != asker_speaker and not is_questionish(canon[k]["text"]):
            out.append(canon[k]["id"])
        k += 1
    return out

def answer_line(seg, is_qseg=False):
    txt = seg["text"]
    if is_qseg and "?" in txt:
        
        tail = txt.split("?")[-1].strip()
        if tail:
            txt = tail
    return f"{seg['speaker']}: {txt}"

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    args = ap.parse_args()

    in_path = Path(args.input) if args.input else sorted([p for p in Path(".").glob("*.json") if not p.name.endswith(".qa.json")])[0]
    client = Groq(api_key=API_KEY)

    raw = load_segments(in_path)
    canon = canonicalize(raw)
    id_to_idx = {s["id"]: i for i,s in enumerate(canon)}
    by_id = {s["id"]: s for s in canon}

    best = {}
    for _,_,txt in chunk_lines(canon):
        out = groq_json(client, PASS1_SYSTEM, f"<TRANSCRIPT_CHUNK>\n{txt}\n</TRANSCRIPT_CHUNK>", PASS1_SCHEMA, "q_detect")
        for q in out.get("questions", []) or []:
            qid = norm_id(q.get("question_id",""))
            if not qid or qid not in id_to_idx:
                continue
            conf = float(q.get("confidence", 0.0))
            if qid not in best or conf > float(best[qid].get("confidence", 0.0)):
                best[qid] = {
                    "question_id": qid,
                    "question_text_clean": str(q.get("question_text_clean","")).strip(),
                    "question_type": str(q.get("question_type","other")).strip(),
                    "is_rhetorical": bool(q.get("is_rhetorical", False)),
                    "confidence": conf,
                }

    def segnum(qid):
        m = re.match(r"#(\d+)", qid)
        return int(m.group(1)) if m else 10**12

    questions = sorted(best.values(), key=lambda x: segnum(x["question_id"]))

    qa_pairs = []
    for q in questions:
        qid = q["question_id"]
        qseg = by_id[qid]
        qtext = q["question_text_clean"] or qseg["text"]
        asker = qseg["speaker"]

        selected = None
        for w in [150, 300, 900]:
            ep = build_episode(canon, id_to_idx, qid, pre_turns=2, window_sec=w, max_turns=120)
            out = groq_json(
                client, PASS2_SYSTEM,
                f"QUESTION_ID: {qid}\nQUESTION_TEXT_CLEAN: {qtext}\n\n<EPISODE>\n{ep}\n</EPISODE>",
                PASS2_SCHEMA, "a_extract"
            )

            status = str(out.get("answer_status","unanswered"))

            evid = []
            for e in (out.get("answer_evidence_ids") or []):
                nid = norm_id(e)
                if nid and nid in by_id:
                    evid.append(nid)

            if not evid and status in ("answered", "partial"):
                evid = pick_reply_evidence(canon, id_to_idx, qid, asker)
                if not evid and (qseg["end"] - qseg["start"]) >= LONG_SEG_SEC:
                    evid = [qid]

            if evid == [qid]:
                repl = pick_reply_evidence(canon, id_to_idx, qid, asker)
                if repl:
                    evid = repl

            evid = expand_evidence(canon, id_to_idx, by_id, qid, evid)
            out["answer_evidence_ids"] = evid
            selected = out

            if status in ("answered","partial") and evid and float(out.get("confidence",0)) >= 0.6:
                break

        if not selected:
            selected = {"answer_status":"unanswered","answer_summary":"","answer_evidence_ids":[],"confidence":0.0,"support_check_passed":False}

        ev = [by_id[eid] for eid in selected["answer_evidence_ids"] if eid in by_id]

        answer_text = "\n".join(answer_line(s, is_qseg=(s["id"]==qid)) for s in ev) if ev else ""

        answerers = sorted({s["speaker"] for s in ev}) if ev else []
        if ev:
            ans_start = min(s["start"] for s in ev); ans_end = max(s["end"] for s in ev)
            latency = max(0.0, ans_start - qseg["start"])
        else:
            ans_start = qseg["end"]; ans_end = qseg["end"]; latency = 0.0

        status = str(selected.get("answer_status","unanswered"))
        if status in ("answered","partial") and not ev:
            status = "unanswered"

        qa_pairs.append({
            "question_id": qid,
            "asker_speaker": asker,
            "question": qtext,
            "question_ts": {"start": qseg["start"], "end": qseg["end"]},
            "question_type": q.get("question_type","other"),
            "is_rhetorical": bool(q.get("is_rhetorical", False)),
            "question_confidence": float(q.get("confidence",0.0)),

            "answer_status": status,
            "answerers_speakers": answerers,
            "answer": answer_text,
            "answer_evidence_ids": [s["id"] for s in ev],
            "answer_ts_range": {"start": ans_start, "end": ans_end},
            "latency_sec": latency,
            "answer_confidence": float(selected.get("confidence",0.0)),
        })

    out = {
        "meeting_file": in_path.name,
        "model": MODEL,
        "canonical_segment_count": len(canon),
        "qa_pair_count": len(qa_pairs),
        "qa_pairs": qa_pairs,
    }
    out_path = in_path.with_suffix("").with_suffix(".qa.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Wrote {out_path}")

if __name__ == "__main__":
    main()
