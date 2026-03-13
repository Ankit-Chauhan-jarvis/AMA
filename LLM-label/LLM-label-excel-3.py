import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter

import pandas as pd
from groq import Groq
from dotenv import load_dotenv

LABEL_SET = ["task", "procedural", "social", "action", "counter"]

LABEL_GUIDE = """
=== ACT4Teams Labeling Guide ===

[task] Problem-Oriented Statements
  Description:
    Professional Competence — all purely technical/content-related statements about
    the topic of the meeting. Questions, sharing of information or knowledge,
    identification of problems and solutions. Often about informing others about
    things or tasks that have happened (not future intentions). Past-oriented.
    Aims/objectives (visions, requirements: "it would be nice if...").
    How should something be? Abstract goals or ideals.
    Suggestion with no concrete action yet (can, maybe, hopefully).
  Does NOT include:
    Statements related to structuring the meeting.
    Statements regarding post-meeting actions.
  Examples:
    "Next one is, remember, you can sign up to be a host on the conference section."
    "So you can make multiple ones there, so that should not be a problem."

[procedural] Procedural Statements
  Description:
    Methodological Competence — everything related to managing or structuring the
    meeting. Goal orientation, time management. Clarification, concretization,
    summarization of contributions. Questions and suggestions about how to proceed
    in the meeting. Present-oriented. Structuring individual contributions.
  Does NOT include:
    Statements about post-meeting actions or tasks.
  Examples:
    Handing over the floor, asking someone to continue.
    "But let me just share my screen here for a second."
    "Evita said this and that, Rebekka said this and that."

[social] Socio-Emotional Statements
  Description:
    Social Competence — everything that contributes to better understanding each
    other and creating a positive atmosphere. Praise, recognition, support.
    Humor, jokes, expressing feelings. Encouraging participation, responding to others.
  Does NOT include:
    Fillers unrelated to previous comments.
  Examples:
    "Yeah, ok, great, alright."

[action] Action-Oriented Statements
  Description:
    Self-Competence & Proactivity — everything that helps advance work after or
    outside the meeting. Action/plan development: concrete steps and strategies.
    Direct task assignment. Taking responsibility. Future-oriented.
  Does NOT include:
    Statements about tasks or behavior during the meeting itself.
    Statements that merely provide information or optional offers.
  Examples:
    "Yeah, I'll do that."
    "By the end of next week I'll have a working version behind Feature Flag."
    "So make sure to read it."
    "I encourage everyone here to step up and proactively shape how we work."

[counter] Counterproductive/Destructive Statements
  Description:
    Negative procedural, socio-emotional, and action-oriented statements —
    everything that harms productivity of the meeting or working relationships.
    Procedural (-): getting lost in unnecessary details.
    Socio-emotional (-): devaluing others.
    Action-oriented (-): no interest in change, responsibility, or action; complaining.
    Interruptions (not letting someone finish).
  Examples:
    "So step one — Sorry Amy, before you go, I just want to..." (interruption)
"""


def build_system_prompt() -> str:
    return (
        "You are labeling meeting transcript sentences for the ACT4Teams research project.\n"
        "Each sentence is one unit of speech from a single speaker.\n"
        f"Choose exactly ONE label from: {', '.join(LABEL_SET)}.\n"
        "Return ONLY valid JSON as instructed — no markdown, no extra text.\n"
        "Short acknowledgements or backchannels (e.g. 'yeah', 'ok', 'great') → label as 'social'.\n"
        "Pick the single best label even when uncertain.\n\n"
        + LABEL_GUIDE
    )


def _extract_json_array(text: str) -> List[Dict[str, Any]]:
    start = text.find("[")
    end   = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON array in model output.")
    return json.loads(text[start: end + 1])


def _call_llm(
    client: Groq,
    model: str,
    system_prompt: str,
    items: List[Tuple[int, str, str]],  # (id, text, speaker)
    temperature: float,
) -> Dict[int, str]:
    """Single LLM call → {id: label}."""
    payload = [
        {"id": sid, "speaker": spk, "text": re.sub(r"\s+", " ", txt).strip()}
        for sid, txt, spk in items
    ]
    user_prompt = (
        f"Label each item with exactly one label from: {', '.join(LABEL_SET)}.\n"
        "The 'speaker' field is context only — do not label by speaker name.\n"
        "Return ONLY a JSON array, no markdown.\n"
        "Return a label for EVERY id, exactly once.\n"
        'Each element: {"id": <int>, "label": "<label>"}\n\n'
        "INPUT:\n" + json.dumps(payload, ensure_ascii=False)
    )
    resp = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        model=model,
        temperature=temperature,
        max_tokens=2000,
    )
    raw = (resp.choices[0].message.content or "").strip()
    result: Dict[int, str] = {}
    for obj in _extract_json_array(raw):
        sid = int(obj["id"])
        lab = str(obj.get("label", "")).strip().lower()
        if lab not in LABEL_SET:
            raise ValueError(f"Invalid label '{lab}' for id={sid}")
        result[sid] = lab
    return result


def label_batch(
    client: Groq,
    model: str,
    system_prompt: str,
    items: List[Tuple[int, str, str]],
    temperature: float = 0.0,
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    """
    Resilient batch labeling:
      1. Retry missing ids up to max_retries times.
      2. If still missing, split batch in half recursively.
      3. Single-item last resort, defaults to 'social' on total failure.
    """
    expected_ids = [sid for sid, _, _ in items]
    id_map       = {sid: (txt, spk) for sid, txt, spk in items}
    collected: Dict[int, str] = {}
    remaining = list(items)

    for attempt in range(1, max_retries + 1):
        try:
            got = _call_llm(client, model, system_prompt, remaining, temperature)
            collected.update({sid: lab for sid, lab in got.items() if sid in id_map})
            missing = [sid for sid in expected_ids if sid not in collected]
            if not missing:
                break
            remaining = [(sid, *id_map[sid]) for sid in missing]
            time.sleep(0.75 * attempt)
        except Exception:
            time.sleep(1.0 * attempt)

    if len(collected) == len(expected_ids):
        return [{"id": sid, "label": collected[sid]} for sid in expected_ids]

    # Split-and-recurse for stubborn items
    still_missing = [(sid, *id_map[sid]) for sid in expected_ids if sid not in collected]
    if len(still_missing) == 1:
        sid, txt, spk = still_missing[0]
        for attempt in range(1, 5):
            try:
                got = _call_llm(client, model, system_prompt, [(sid, txt, spk)], temperature)
                if sid in got:
                    collected[sid] = got[sid]
                    break
            except Exception:
                time.sleep(1.0 * attempt)
        collected.setdefault(sid, "social")
    else:
        mid = len(still_missing) // 2
        for item in label_batch(client, model, system_prompt, still_missing[:mid],  temperature, max_retries):
            collected[item["id"]] = item["label"]
        for item in label_batch(client, model, system_prompt, still_missing[mid:], temperature, max_retries):
            collected[item["id"]] = item["label"]

    return [{"id": sid, "label": collected.get(sid, "social")} for sid in expected_ids]



_ACK_TOKENS = {
    "ok", "okay", "alright", "right", "cool", "great", "nice",
    "yeah", "yep", "yup", "sure", "thanks", "thank", "perfect",
    "awesome", "fine", "good", "exactly", "indeed",
    "mm", "mhm", "hmm", "uh", "um", "hi", "hey",
}
_DISCOURSE_PREFIX = {"so", "well", "um", "uh", "like", "oh", "and", "anyway"}
_ACK_PHRASES      = {("all", "right"), ("sounds", "good"), ("for", "sure"), ("thank", "you")}


def _is_backchannel(text: str) -> bool:
    s = text.strip()
    if not s:               return True
    if "?" in s:            return False
    if re.search(r"\d", s): return False
    norm = re.sub(r"[^\w\s']+", "", s.lower()).strip()
    toks = norm.split()
    while toks and toks[0] in _DISCOURSE_PREFIX:
        toks = toks[1:]
    if not toks:                                                return True
    if len(toks) <= 3 and all(t in _ACK_TOKENS for t in toks): return True
    if tuple(toks) in _ACK_PHRASES:                            return True
    return False


def parse_transcript(json_path: str) -> List[Dict[str, Any]]:
    """
    Parses WhisperX JSON and returns sentence rows:
      [{"id":1, "text":"...", "start_time":2.9, "end_time":6.6, "speaker":"SPEAKER_04"}, ...]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Accept both {"segments": [...]} and bare list formats
    segments = data.get("segments", data) if isinstance(data, dict) else data

    rows: List[Dict[str, Any]] = []
    next_id = 1

    for seg in segments:
        text    = str(seg.get("text", "") or "").strip()
        speaker = str(seg.get("speaker", "") or "").strip()

        if not text or _is_backchannel(text):
            continue

        try:
            start = float(seg["start"])
            end   = float(seg["end"])
        except (KeyError, TypeError, ValueError):
            continue

        rows.append({
            "id":         next_id,
            "text":       text,
            "start_time": round(start, 3),
            "end_time":   round(end,   3),
            "speaker":    speaker,
        })
        next_id += 1

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentence-level ACT4Teams labeling via Groq.")
    parser.add_argument("--transcript_json", required=True)
    parser.add_argument("--out",        default="labeled_sentences.xlsx")
    parser.add_argument("--batch_size", type=int, default=20)
    args = parser.parse_args()

    if not Path(args.transcript_json).exists():
        raise FileNotFoundError(f"Not found: {args.transcript_json}")

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env")

    client = Groq(api_key=api_key)
    model  = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    system_prompt = build_system_prompt()

    # ── Parse ─────────────────────────────────────────────────────────────────
    print("Parsing transcript...")
    rows = parse_transcript(args.transcript_json)
    print(f"  {len(rows)} sentences after backchannel filtering.")
    for spk, cnt in sorted(Counter(r["speaker"] for r in rows).items()):
        print(f"  {spk}: {cnt} sentences")

    # ── Label ─────────────────────────────────────────────────────────────────
    items: List[Tuple[int, str, str]] = [(r["id"], r["text"], r["speaker"]) for r in rows]
    id_to_label: Dict[int, str] = {}
    total_batches = (len(items) + args.batch_size - 1) // args.batch_size
    print(f"\nLabeling {len(items)} sentences across {total_batches} batches...")

    for batch_num, start in enumerate(range(0, len(items), args.batch_size), 1):
        batch = items[start: start + args.batch_size]
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} sentences)...", end=" ", flush=True)
        for item in label_batch(client, model, system_prompt, batch):
            id_to_label[item["id"]] = item["label"]
        print("done")

    # ── Write Excel ───────────────────────────────────────────────────────────
    out_rows = [
        {
            "Id":         r["id"],
            "text":       r["text"],
            "start_time": r["start_time"],
            "end_time":   r["end_time"],
            "label":      id_to_label.get(r["id"], ""),
            "speaker":    r["speaker"],
        }
        for r in rows
    ]
    df = pd.DataFrame(out_rows, columns=["Id", "text", "start_time", "end_time", "label", "speaker"])
    df.to_excel(args.out, index=False)

    print(f"\nWrote {len(df)} labeled sentences → {args.out}")
    print("\nLabel distribution:")
    print(df["label"].value_counts().to_string())
    print("\nSentences per speaker:")
    print(df["speaker"].value_counts().to_string())


if __name__ == "__main__":
    main()