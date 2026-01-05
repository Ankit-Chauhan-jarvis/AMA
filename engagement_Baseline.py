import json
import re
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Config / heuristics
# =========================

# Short acknowledgements = passive (verbal attention but low content)
BACKCHANNELS = {
    "yeah", "yes", "yep", "no", "nope", "okay", "ok", "okey",
    "mm", "mmhmm", "mhm", "uh-huh", "uh huh", "uh",
    "right", "sure", "cool", "great", "thanks", "thank you",
    "alright", "all right",
}

# Question words / auxiliaries (used carefully)
QUESTION_WORDS = {"who", "what", "when", "where", "why", "how", "which"}
AUX_START = {"is", "are", "am", "was", "were", "do", "does", "did",
             "can", "could", "should", "would", "will", "have", "has", "had"}

# Action / commitment / planning markers (active)
ACTION_RE = re.compile(
    r"\b("
    r"i'?ll|i will|we will|let's|we should|we need to|need to|plan to|going to|"
    r"i can|we can|follow up|next step|"
    r"by (monday|tuesday|wednesday|thursday|friday|tomorrow|next week)|"
    r"assign|take care of"
    r")\b",
    re.IGNORECASE
)

# Thresholds (tune later)
MIN_PASSIVE_WORDS = 3      # <= this and short duration => likely passive
MIN_ACTIVE_WORDS = 8       # >= this => likely active
MIN_ACTIVE_SEC = 2.5       # >= this => likely active
SHORT_UTTERANCE_SEC = 2.0  # if short duration and short text => passive


# =========================
# Helpers
# =========================

def normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())

def word_count(text: str) -> int:
    return len(normalize(text).split())

def is_backchannel(text: str) -> bool:
    t = normalize(text)
    if not t:
        return True
    w = t.split()
    # "yeah" / "ok" / "mmhmm" etc.
    if len(w) <= 2 and t in BACKCHANNELS:
        return True
    if len(w) == 1 and w[0] in BACKCHANNELS:
        return True
    return False

def is_question(text: str) -> bool:
    """
    Baseline question detector.
    - If contains '?': question
    - If starts like a question and looks addressed ("you/we/it/this/that"): question
    - If contains common question phrases: question
    """
    raw = (text or "").strip()
    t = normalize(text)
    if not t:
        return False

    if "?" in raw:
        return True

    tokens = t.split()
    first = tokens[0] if tokens else ""

    # Avoid false positives like "when I joined..." by requiring pronoun later
    if first in QUESTION_WORDS or first in AUX_START:
        if len(tokens) >= 3 and re.search(r"\b(you|we|it|this|that)\b", t):
            return True

    # Common question phrases
    if re.search(r"\b(quick question|any questions|do you|can you|could you|would you|"
                 r"does it|does this|does that)\b", t):
        return True

    return False

def classify_engagement(text: str, duration_sec: float) -> str:
    """
    Returns: "active" or "passive" for a single utterance.
    Baseline logic:
      - backchannels => passive
      - questions => active
      - action/commitment markers => active
      - very short + few words => passive
      - long / contentful => active
      - else => passive
    """
    wc = word_count(text)

    if is_backchannel(text):
        return "passive"

    if is_question(text):
        return "active"

    if ACTION_RE.search(text or ""):
        return "active"

    if wc <= MIN_PASSIVE_WORDS and duration_sec < SHORT_UTTERANCE_SEC:
        return "passive"

    if wc >= MIN_ACTIVE_WORDS or duration_sec >= MIN_ACTIVE_SEC:
        return "active"

    return "passive"


# =========================
# Loading & alignment
# =========================

def load_diarization_txt(path: str):
    """
    Expected line format:
      start=3.846 stop=10.161 speaker=SPEAKER_00
    Returns sorted list of (start, end, speaker).
    """
    diar = []
    pat = re.compile(
        r"start=(?P<start>\d+(\.\d+)?)\s+stop=(?P<stop>\d+(\.\d+)?)\s+speaker=(?P<spk>\S+)"
    )
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = pat.match(line)
            if not m:
                continue
            diar.append((float(m.group("start")), float(m.group("stop")), m.group("spk")))
    diar.sort(key=lambda x: x[0])
    return diar

def load_whisper_segments(path: str):
    """
    transcript_detailed.json expected to have top-level "segments" list (Whisper format).
    Each segment has start/end/text.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments = data.get("segments", [])
    out = []
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        text = (seg.get("text", "") or "").strip()
        if not text:
            continue
        out.append((start, end, text))
    return out

def assign_speaker_to_segment(seg_start: float, seg_end: float, diar):
    """
    Assign speaker by maximum overlap between a Whisper segment and diarization segments.
    """
    best_spk = "UNKNOWN"
    best_olap = 0.0

    for ds, de, spk in diar:
        if de <= seg_start:
            continue
        if ds >= seg_end:
            break
        overlap = max(0.0, min(seg_end, de) - max(seg_start, ds))
        if overlap > best_olap:
            best_olap = overlap
            best_spk = spk

    return best_spk


# =========================
# Main computation
# =========================

def compute_engagement(diar_path: str, transcript_path: str):
    diar = load_diarization_txt(diar_path)
    whisper = load_whisper_segments(transcript_path)

    utterances = []
    for start, end, text in whisper:
        spk = assign_speaker_to_segment(start, end, diar)
        utterances.append({
            "speaker": spk,
            "start": start,
            "end": end,
            "duration": max(0.0, end - start),
            "text": text,
            "words": word_count(text),
        })

    # Per-speaker aggregation
    metrics = defaultdict(lambda: {
        "active_time": 0.0, "passive_time": 0.0,
        "active_turns": 0, "passive_turns": 0,
        "active_words": 0, "passive_words": 0,
        "total_time": 0.0, "total_turns": 0, "total_words": 0
    })

    total_active_time = 0.0
    total_passive_time = 0.0

    for u in utterances:
        label = classify_engagement(u["text"], u["duration"])
        spk = u["speaker"]
        m = metrics[spk]

        m["total_time"] += u["duration"]
        m["total_turns"] += 1
        m["total_words"] += u["words"]

        if label == "active":
            m["active_time"] += u["duration"]
            m["active_turns"] += 1
            m["active_words"] += u["words"]
            total_active_time += u["duration"]
        else:
            m["passive_time"] += u["duration"]
            m["passive_turns"] += 1
            m["passive_words"] += u["words"]
            total_passive_time += u["duration"]

    rows = []
    for spk, m in metrics.items():
        tot_time = m["total_time"] if m["total_time"] > 0 else 1e-9
        active_ratio = m["active_time"] / tot_time
        rows.append({
            "speaker": spk,
            "total_time_sec": m["total_time"],
            "active_time_sec": m["active_time"],
            "passive_time_sec": m["passive_time"],
            "total_turns": m["total_turns"],
            "active_turns": m["active_turns"],
            "passive_turns": m["passive_turns"],
            "total_words": m["total_words"],
            "active_words": m["active_words"],
            "passive_words": m["passive_words"],
            "active_ratio": active_ratio,
            "passive_ratio": 1.0 - active_ratio,
        })

    df = pd.DataFrame(rows).sort_values("total_time_sec", ascending=False).reset_index(drop=True)

    meeting_total = total_active_time + total_passive_time
    meeting_summary = {
        "meeting_active_time_sec": total_active_time,
        "meeting_passive_time_sec": total_passive_time,
        "meeting_total_time_sec": meeting_total,
        "meeting_active_pct": (total_active_time / meeting_total * 100.0) if meeting_total > 0 else 0.0,
        "meeting_passive_pct": (total_passive_time / meeting_total * 100.0) if meeting_total > 0 else 0.0,
    }

    return df, meeting_summary


def plot_active_ratio_bar(df: pd.DataFrame, title: str = "Active engagement ratio per speaker"):
    speakers = df["speaker"].tolist()
    ratios = df["active_ratio"].tolist()

    plt.figure(figsize=(8, 4))
    plt.bar(speakers, ratios)
    plt.ylabel("Active ratio (active_time / total_time)")
    plt.title(title)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_meeting_split(meeting_summary: dict, title: str = "Meeting-level active vs passive split"):
    labels = ["Active", "Passive"]
    values = [meeting_summary["meeting_active_time_sec"], meeting_summary["meeting_passive_time_sec"]]

    plt.figure(figsize=(5, 4))
    plt.bar(labels, values)
    plt.ylabel("Time (sec)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def main():
    diar_path = "diarization.txt"
    transcript_path = "transcript_detailed.json"

    df, meeting = compute_engagement(diar_path, transcript_path)

    # Print table
    print("\n=== Per-speaker engagement table ===\n")
    show = df.copy()
    for c in ["total_time_sec", "active_time_sec", "passive_time_sec", "active_ratio", "passive_ratio"]:
        if c in show.columns:
            show[c] = show[c].round(3)
    print(show.to_string(index=False))

    # Meeting-level summary
    print("\n=== Meeting-level active/passive split ===")
    print(f"Active time (sec):  {meeting['meeting_active_time_sec']:.2f}  ({meeting['meeting_active_pct']:.2f}%)")
    print(f"Passive time (sec): {meeting['meeting_passive_time_sec']:.2f}  ({meeting['meeting_passive_pct']:.2f}%)")
    print(f"Total time (sec):   {meeting['meeting_total_time_sec']:.2f}")

    # Plots
    plot_active_ratio_bar(df)
    plot_meeting_split(meeting)


if __name__ == "__main__":
    main()
