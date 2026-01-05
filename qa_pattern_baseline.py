import json
import re
import sys
from typing import List, Dict, Any, Optional

# ---------------- Question / answer heuristics ---------------- #

QUESTION_PHRASES = [
    "i have a question",
    "i had a question",
    "quick question",
    "any questions",
    "any question",
    "my question is",
    "do you have any questions",
    "are there any questions",
    "does that make sense",
    "does this make sense",
    "does it make sense",
    "the next question",
    "next question",
]

META_QUESTION_PHRASES = [
    "any questions",
    "any question",
    "does that make sense",
    "does this make sense",
    "does it make sense",
    "make no sense",
    "like to bring up for discussion",
    "anything else",
    "any other business",
]

BACKCHANNELS = {
    "yeah", "yes", "yep", "no", "nope", "okay", "ok", "okey",
    "mm", "mmhmm", "mhm", "uh-huh", "uh huh",
    "right", "sure", "exactly", "great", "cool", "nice",
    "thanks", "thank you",
}


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def split_into_sentences(text: str) -> List[str]:
    """
    Very simple sentence splitter based on punctuation.
    """
    text = text.strip()
    if not text:
        return []
    parts = re.split(r'(?<=[\.\?\!])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def sentence_is_question(sentence: str) -> bool:
    """
    Decide if a *single sentence* is a question.

    Conservative rule:
      - Ends with '?', OR
      - Contains one of the explicit QUESTION_PHRASES.
    """
    s_orig = sentence.strip()
    s_norm = normalize_text(s_orig)

    if not s_norm:
        return False

    # 1) Ends with a question mark
    if s_orig.endswith("?"):
        return True

    # 2) Explicit phrases like "quick question", "the next question", etc.
    for phrase in QUESTION_PHRASES:
        if phrase in s_norm:
            return True

    return False


def is_meta_question_sentence(sentence: str) -> bool:
    """
    Detect "meta" questions like "any questions?", "does it make sense?",
    "like to bring up for discussion?", etc.
    """
    s_norm = normalize_text(sentence)
    for phrase in META_QUESTION_PHRASES:
        if phrase in s_norm:
            return True
    return False


def turn_question_ratio(text: str) -> float:
    """
    Fraction of sentences in a turn that look like questions.
    """
    sents = split_into_sentences(text)
    if not sents:
        return 0.0
    num_q = sum(1 for s in sents if sentence_is_question(s))
    return num_q / len(sents)


def is_question_turn(text: str) -> bool:
    """
    Turn-level question detector.
    """
    return turn_question_ratio(text) > 0.0


def extract_question_sentences(full_text: str) -> str:
    """
    From a long turn, extract only the sentences that look like questions.
    If none found, return the original text (fallback).
    """
    sents = split_into_sentences(full_text)
    question_sents = [s for s in sents if sentence_is_question(s)]

    if not question_sents:
        return full_text.strip()

    # Keep all question sentences in order
    return " ".join(question_sents).strip()


def is_backchannel(text: str) -> bool:
    """
    Detect short acknowledgement / backchannel utterances.
    """
    if not text or not text.strip():
        return True

    text_norm = normalize_text(text)
    words = text_norm.split()

    # Very short utterances are often backchannels
    if len(words) <= 2 and text_norm in BACKCHANNELS:
        return True

    if len(words) == 1 and text_norm in BACKCHANNELS:
        return True

    return False


# ---------------- Core Q-A detection ---------------- #

def load_segments_from_json(path: str) -> List[Dict[str, Any]]:
    """
    Load transcript segments from JSON.
    Assumes either:
      - a list of segments, or
      - a dict with a 'segments' field containing the list.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "segments" in data:
        return data["segments"]
    else:
        raise ValueError("Unexpected JSON structure. Adjust load_segments_from_json().")


def detect_questions(segments: List[Dict[str, Any]]) -> List[int]:
    """
    Return indices of segments that contain at least one question sentence.
    """
    question_indices = []
    for i, seg in enumerate(segments):
        text = seg.get("text", "") or ""
        if is_question_turn(text):
            question_indices.append(i)
    return question_indices


def find_answer_for_question(
    segments: List[Dict[str, Any]],
    q_index: int,
    max_answer_gap_sec: float = 90.0,
    min_answer_words: int = 3,
) -> Optional[Dict[str, Any]]:
    """
    Given an index of a question segment, search forward for an answer.

    Strategy:
      - Look at segments after the question.
      - Stop if we exceed max_answer_gap_sec in time.
      - Candidate answers:
          - Different speaker than question.
          - Not a backchannel.
          - At least min_answer_words.
          - Not a *question-heavy* turn (question_ratio >= 0.5).
      - Take the first suitable candidate and extend it with subsequent
        segments from the same speaker that are close in time.
    """
    q_seg = segments[q_index]
    q_speaker = q_seg.get("speaker")
    q_end = float(q_seg.get("end", q_seg.get("start", 0.0)))

    first_answer_idx: Optional[int] = None

    for j in range(q_index + 1, len(segments)):
        seg = segments[j]
        start = float(seg.get("start", 0.0))
        text = seg.get("text", "") or ""
        speaker = seg.get("speaker")

        # Stop if too far in time
        if start - q_end > max_answer_gap_sec:
            break

        text_norm = normalize_text(text)
        words = text_norm.split()

        # We want an answer from SOMEONE ELSE
        if speaker == q_speaker:
            continue

        # Skip backchannels
        if is_backchannel(text):
            continue

        # Skip very short utterances
        if len(words) < min_answer_words:
            continue

        # Skip segments that are themselves mostly questions
        if turn_question_ratio(text) >= 0.5:
            continue

        # We found a candidate
        first_answer_idx = j
        break

    if first_answer_idx is None:
        return None

    # Extend answer with subsequent turns from same speaker
    answer_speaker = segments[first_answer_idx].get("speaker")
    answer_start = float(segments[first_answer_idx].get("start", 0.0))
    answer_end = float(segments[first_answer_idx].get("end", answer_start))
    answer_text_parts = [segments[first_answer_idx].get("text", "") or ""]

    for k in range(first_answer_idx + 1, len(segments)):
        seg = segments[k]
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        text = seg.get("text", "") or ""
        speaker = seg.get("speaker")

        if start - q_end > max_answer_gap_sec:
            break

        if speaker != answer_speaker:
            break

        answer_text_parts.append(text)
        answer_end = end

    answer_text = " ".join(answer_text_parts).strip()

    return {
        "answer_speaker": answer_speaker,
        "answer_start": answer_start,
        "answer_end": answer_end,
        "answer_text": answer_text,
    }


def detect_qa_pairs(
    segments: List[Dict[str, Any]],
    max_answer_gap_sec: float = 90.0,
    min_answer_words: int = 3,
) -> List[Dict[str, Any]]:
    """
    Detect question–answer pairs across the whole meeting.
    Returns a list of dicts, each containing question and answer info.
    """
    qa_pairs = []
    question_indices = detect_questions(segments)

    for q_idx in question_indices:
        q_seg = segments[q_idx]
        full_question_text = q_seg.get("text", "") or ""

        sents = split_into_sentences(full_question_text)
        question_sents = [s for s in sents if sentence_is_question(s)]

        if question_sents:
            question_text_clean = " ".join(question_sents).strip()
        else:
            question_text_clean = full_question_text.strip()

        # Meta-only questions (e.g. "any questions? does this make sense?")
        meta_only = bool(question_sents) and all(
            is_meta_question_sentence(s) for s in question_sents
        )

        if meta_only:
            answer = None
        else:
            # >>> fixed typo here: use q_idx, not q_index
            answer = find_answer_for_question(
                segments,
                q_idx,
                max_answer_gap_sec=max_answer_gap_sec,
                min_answer_words=min_answer_words,
            )

        qa_pairs.append({
            "question_index": q_idx,
            "question_speaker": q_seg.get("speaker"),
            "question_start": float(q_seg.get("start", 0.0)),
            "question_end": float(q_seg.get("end", 0.0)),
            "question_text": question_text_clean,
            "meta_only": meta_only,
            "answer": answer,   # may be None if unanswered
        })

    return qa_pairs


# ---------------- Pretty printing ---------------- #

def print_qa_pairs(qa_pairs: List[Dict[str, Any]]):
    """
    Print Q-A pairs in a readable format on the console.
    """
    if not qa_pairs:
        print("No questions detected.")
        return

    for i, qa in enumerate(qa_pairs, start=1):
        print("=" * 80)
        print(f"Q{i}")
        print("- Question speaker: {qa_speaker} (index {q_idx})".format(
            qa_speaker=qa["question_speaker"],
            q_idx=qa["question_index"],
        ))
        print("  Question start: {:.2f}s, end: {:.2f}s".format(
            qa["question_start"], qa["question_end"]
        ))
        print("  Question text: {}".format(qa["question_text"]))
        if qa["meta_only"]:
            print("  [META CHECK-IN QUESTION]")
        ans = qa["answer"]
        if ans is None:
            print("  Answer: [NO CLEAR ANSWER FOUND]")
        else:
            print("\n  Answer speaker: {}".format(ans["answer_speaker"]))
            print("  Answer start: {:.2f}s, end: {:.2f}s".format(
                ans["answer_start"], ans["answer_end"]
            ))
            text = ans["answer_text"].strip()
            if len(text) > 400:
                preview = text[:400] + " ..."
            else:
                preview = text
            print("  Answer text: {}".format(preview))
        print()


# ---------------- Main ---------------- #

def main(json_path: str):
    segments = load_segments_from_json(json_path)
    qa_pairs = detect_qa_pairs(segments)

    print("\nDetected {} question(s).\n".format(len(qa_pairs)))
    print_qa_pairs(qa_pairs)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python qa_pattern_baseline.py path/to/your.json")
        sys.exit(1)

    path = sys.argv[1]
    main(path)
