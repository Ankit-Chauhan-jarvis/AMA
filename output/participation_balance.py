import json
from collections import defaultdict
import math
import sys

import matplotlib.pyplot as plt
import pandas as pd


# ---------- Utility functions ----------

def compute_gini(values):
    """
    Compute Gini coefficient for a list of non-negative values.
    0 = perfectly equal, 1 = perfectly unequal.
    """
    # Filter out negatives (if any) and handle edge cases
    values = [v for v in values if v >= 0]
    if not values:
        return 0.0

    total = sum(values)
    if total == 0:
        return 0.0

    # Sort ascending
    values_sorted = sorted(values)
    n = len(values_sorted)

    # Gini formula: G = (2 * sum(i * x_i)) / (n * sum x) - (n + 1) / n
    cumulative = 0.0
    for i, x in enumerate(values_sorted, start=1):
        cumulative += i * x

    gini = (2.0 * cumulative) / (n * total) - (n + 1.0) / n
    return gini


def compute_hhi(shares):
    """
    Compute Herfindahl-Hirschman Index (HHI) from shares that sum to 1.
    For speaking time shares, higher = more concentrated (less balanced).
    """
    return sum(s ** 2 for s in shares)


# ---------- Core computation ----------

def compute_participation_metrics(segments):
    """
    segments: list of dicts with keys:
      - 'speaker'
      - 'start' (seconds, float)
      - 'end'   (seconds, float)
      - 'text'  (string)

    Returns:
      - df: pandas DataFrame with per-speaker metrics
      - global_metrics: dict with Gini, HHI, etc.
    """
    durations = defaultdict(float)
    word_counts = defaultdict(int)
    turn_counts = defaultdict(int)

    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        text = seg.get("text", "") or ""

        duration = max(0.0, end - start)
        durations[speaker] += duration
        turn_counts[speaker] += 1

        # Basic word count (you can replace with a smarter tokenizer if needed)
        words = text.strip().split()
        word_counts[speaker] += len(words)

    # Totals
    total_time = sum(durations.values())
    total_words = sum(word_counts.values()) or 1  # avoid division by zero

    # Build DataFrame
    data = []
    for speaker in sorted(durations.keys()):
        time_sec = durations[speaker]
        time_pct = (time_sec / total_time * 100.0) if total_time > 0 else 0.0

        words = word_counts[speaker]
        words_pct = (words / total_words * 100.0) if total_words > 0 else 0.0

        turns = turn_counts[speaker]
        avg_words_per_turn = words / turns if turns > 0 else 0.0

        data.append({
            "speaker": speaker,
            "time_sec": time_sec,
            "time_pct": time_pct,
            "words": words,
            "words_pct": words_pct,
            "turns": turns,
            "avg_words_per_turn": avg_words_per_turn,
        })

    df = pd.DataFrame(data)
    # Sort by speaking time descending for nicer display
    df = df.sort_values(by="time_sec", ascending=False).reset_index(drop=True)

    # Global balance metrics (using *time* as primary indicator)
    time_values = [durations[s] for s in durations]
    time_shares = [
        (durations[s] / total_time) if total_time > 0 else 0.0
        for s in durations
    ]

    gini_time = compute_gini(time_values)
    hhi_time = compute_hhi(time_shares)

    global_metrics = {
        "total_time_sec": total_time,
        "total_words": total_words,
        "gini_time": gini_time,
        "hhi_time": hhi_time,
    }

    return df, global_metrics


def plot_speaking_time_bar(df, title="Speaking time by participant"):
    """
    Create a bar chart of speaking time (%) per participant using the DataFrame.
    """
    speakers = df["speaker"].tolist()
    time_pct = df["time_pct"].tolist()

    plt.figure(figsize=(8, 4))
    plt.bar(speakers, time_pct)
    plt.ylabel("Speaking time (%)")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


# ---------- Loading & main ----------

def load_segments_from_json(path):
    """
    Loads your JSON file.
    Assumes it's a list of turn dicts.
    If your file structure is different, adjust this function.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # If your JSON is directly a list of segments, just return it.
    # Otherwise, you may need to access a field, e.g. data["segments"].
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "segments" in data:
        return data["segments"]
    else:
        raise ValueError("Unexpected JSON structure. Adjust load_segments_from_json().")


def main(json_path):
    segments = load_segments_from_json(json_path)
    df, global_metrics = compute_participation_metrics(segments)

    print("\n=== Participation metrics per speaker ===\n")
    # Pretty print with rounded numbers
    display_df = df.copy()
    for col in ["time_sec", "time_pct", "words_pct", "avg_words_per_turn"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(2)
    print(display_df.to_string(index=False))

    print("\n=== Global participation balance metrics ===")
    print(f"Total speaking time (sec): {global_metrics['total_time_sec']:.2f}")
    print(f"Total words: {global_metrics['total_words']}")
    print(f"Gini (time): {global_metrics['gini_time']:.4f}")
    print(f"HHI (time):  {global_metrics['hhi_time']:.4f}")
    print("Note: higher Gini/HHI = more concentrated / less balanced participation")

    # Plot bar chart
    plot_speaking_time_bar(df)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python participation_balance.py path/to/your.json")
        sys.exit(1)

    json_path = sys.argv[1]
    main(json_path)
