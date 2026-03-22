
import json
import re
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter


TRACK_BLUE = "#dce8f5"


def sec_to_label(sec: float) -> str:
    sec = float(sec)
    total_seconds = int(round(sec))
    m, s = divmod(total_seconds, 60)
    return f"{m}m {s:02d}s" if m > 0 else f"{s}s"


def gantt_fmt_factory(base_time: datetime):
    def gantt_fmt(x, pos=None):
        dt = mdates.num2date(x).replace(tzinfo=None)
        total_seconds = int(round((dt - base_time).total_seconds()))
        total_seconds = max(total_seconds, 0)
        m, s = divmod(total_seconds, 60)
        return f"{m:02d}:{s:02d}"
    return gantt_fmt


def speaker_sort_key(speaker_name: str):
    match = re.search(r"(\d+)$", str(speaker_name))
    if match:
        return (0, int(match.group(1)))
    if str(speaker_name) == "UNKNOWN":
        return (2, 9999)
    return (1, str(speaker_name))


def load_meeting_json(json_path: Path) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for item in data:
        start = float(item["start"])
        end = float(item["end"])
        if end < start:
            start, end = end, start

        rows.append(
            {
                "speaker": item["speaker"],
                "start": start,
                "end": end,
                "duration": max(end - start, 0.0),
                "text": item.get("text", ""),
            }
        )

    return pd.DataFrame(rows).sort_values(["start", "end"]).reset_index(drop=True)


def overlap_duration(seg_start: float, seg_end: float, q_start: float, q_end: float) -> float:
    return max(0.0, min(seg_end, q_end) - max(seg_start, q_start))


def quarter_name(idx: int) -> str:
    names = {
        1: "first quarter",
        2: "second quarter",
        3: "third quarter",
        4: "fourth quarter",
    }
    return names[idx]


def join_labels(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def build_quarter_analysis(df: pd.DataFrame) -> tuple[list[dict], dict]:
    meeting_start = float(df["start"].min())
    meeting_end = float(df["end"].max())
    meeting_total = meeting_end - meeting_start

    if meeting_total <= 0:
        raise ValueError("Meeting duration must be greater than zero.")

    quarter_len = meeting_total / 4.0
    speakers = sorted(df["speaker"].dropna().unique().tolist(), key=speaker_sort_key)
    total_turns_meeting = len(df)

    quarter_stats = []
    phase_totals = []
    phase_turn_totals = []
    cumulative_seen_speakers = set()

    for i in range(4):
        q_idx = i + 1
        q_start = meeting_start + i * quarter_len
        q_end = meeting_start + (i + 1) * quarter_len if i < 3 else meeting_end

        speaker_time = {speaker: 0.0 for speaker in speakers}
        speaker_turns = {speaker: 0 for speaker in speakers}
        active_speakers = set()

        for row in df.itertuples(index=False):
            ov = overlap_duration(float(row.start), float(row.end), q_start, q_end)
            if ov > 0:
                speaker_time[row.speaker] += ov
                speaker_turns[row.speaker] += 1
                active_speakers.add(row.speaker)

        total_time = sum(speaker_time.values())
        total_turns = sum(speaker_turns.values())
        active_speaker_count = len(active_speakers)
        active_ratio = active_speaker_count / len(speakers) if speakers else 0.0

        sorted_speakers = sorted(speaker_time.items(), key=lambda x: x[1], reverse=True)
        dominant_speaker = None
        dominant_share = 0.0
        second_speaker = None
        second_share = 0.0
        top2_concentration = 0.0
        dominance_strength = 0.0

        if total_time > 0 and sorted_speakers:
            dominant_speaker = sorted_speakers[0][0]
            dominant_share = sorted_speakers[0][1] / total_time

            if len(sorted_speakers) > 1:
                second_speaker = sorted_speakers[1][0]
                second_share = sorted_speakers[1][1] / total_time

            top2_concentration = dominant_share + second_share
            dominance_strength = dominant_share - second_share

        speaker_count_threshold_flag = active_speaker_count <= 2
        turn_density = total_turns / (quarter_len / 60.0) if quarter_len > 0 else 0.0
        new_entries = active_speakers - cumulative_seen_speakers
        speaker_entry_count = len(new_entries)
        cumulative_seen_speakers.update(active_speakers)

        phase_totals.append(total_time)
        phase_turn_totals.append(total_turns)

        quarter_stats.append(
            {
                "quarter_index": q_idx,
                "quarter_name": quarter_name(q_idx),
                "start_sec": q_start,
                "end_sec": q_end,
                "total_time": total_time,
                "total_turns": total_turns,
                "active_speakers": sorted(active_speakers, key=speaker_sort_key),
                "active_speaker_count": active_speaker_count,
                "active_ratio": active_ratio,
                "speaker_time": speaker_time,
                "speaker_turns": speaker_turns,
                "dominant_speaker": dominant_speaker,
                "dominant_share": dominant_share,
                "second_speaker": second_speaker,
                "second_share": second_share,
                "top2_concentration": top2_concentration,
                "speaker_count_threshold_flag": speaker_count_threshold_flag,
                "dominance_strength": dominance_strength,
                "turn_density": turn_density,
                "speaker_entry_count": speaker_entry_count,
            }
        )

    overall = {
        "meeting_start": meeting_start,
        "meeting_end": meeting_end,
        "meeting_total": meeting_total,
        "meeting_total_turns": total_turns_meeting,
        "phase_totals": phase_totals,
        "phase_shares": [x / meeting_total if meeting_total > 0 else 0.0 for x in phase_totals],
        "phase_turn_totals": phase_turn_totals,
        "phase_turn_shares": [x / total_turns_meeting if total_turns_meeting > 0 else 0.0 for x in phase_turn_totals],
        "speakers": speakers,
    }

    for q, share_time, share_turns in zip(quarter_stats, overall["phase_shares"], overall["phase_turn_shares"]):
        q["phase_share_of_meeting_time"] = share_time
        q["phase_share_of_total_turns"] = share_turns

    return quarter_stats, overall


def is_speaker_dominated(q: dict) -> bool:
    if q["total_time"] <= 0:
        return False
    return q["dominant_share"] > 0.50


def classify_distribution(quarter_stats: list[dict], overall: dict) -> dict:
    active_ok_flags = [q["active_ratio"] >= 0.5 for q in quarter_stats]
    dominated_quarters = [q for q in quarter_stats if is_speaker_dominated(q)]

    low_activity_quarters = [
        q["quarter_name"]
        for q, share in zip(quarter_stats, overall["phase_shares"])
        if share < 0.20
    ]

    fixed_description = (
        "This Gantt chart shows who spoke when and for how long during the meeting. "
        "It helps identify whether speaking turns were evenly distributed over time and among speakers, "
        "whether some participants dominated specific phases, and whether some phases showed higher or lower speaking activity."
    )

    if len(dominated_quarters) >= 2:
        q_names = [q["quarter_name"] for q in dominated_quarters]
        interpretation = (
            f"In this particular meeting, speaking time was unevenly distributed across meeting participants "
            f"for the {join_labels(q_names)}, indicating that the meeting was led by one or a few central speakers in these phases."
        )
        return {
            "classification": "uneven_across_speakers",
            "fixed_description": fixed_description,
            "adaptive_interpretation": interpretation,
        }

    if len(low_activity_quarters) >= 1:
        if len(low_activity_quarters) == 1:
            interpretation = (
                f"In this particular meeting, speaking time was concentrated outside the {low_activity_quarters[0]}, "
                f"indicating that overall participation occurred in bursts rather than continuously."
            )
        else:
            interpretation = (
                f"In this particular meeting, speaking time was concentrated outside the {join_labels(low_activity_quarters)}, "
                f"indicating that overall participation occurred in bursts rather than continuously."
            )
        return {
            "classification": "uneven_across_time",
            "fixed_description": fixed_description,
            "adaptive_interpretation": interpretation,
        }

    if all(active_ok_flags) and len(dominated_quarters) == 0:
        interpretation = (
            "In this particular meeting, speaking time was evenly distributed across the course of the meeting, "
            "indicating a generally stable pattern of participation over time and speakers."
        )
        return {
            "classification": "even_distribution",
            "fixed_description": fixed_description,
            "adaptive_interpretation": interpretation,
        }

    if len(dominated_quarters) == 1:
        q_name = dominated_quarters[0]["quarter_name"]
        interpretation = (
            f"In this particular meeting, speaking time was unevenly distributed across meeting participants "
            f"for the {q_name}, indicating that one speaker temporarily dominated this phase."
        )
        return {
            "classification": "single_phase_speaker_dominance",
            "fixed_description": fixed_description,
            "adaptive_interpretation": interpretation,
        }

    interpretation = (
        "In this particular meeting, speaking time showed a mixed distribution pattern, "
        "indicating moderate variation in participation across meeting phases and speakers."
    )
    return {
        "classification": "mixed_pattern",
        "fixed_description": fixed_description,
        "adaptive_interpretation": interpretation,
    }


def save_analysis_outputs(
    json_path: Path,
    quarter_stats: list[dict],
    overall: dict,
    interpretation: dict,
    output_txt: str | None = None,
    output_json: str | None = None,
) -> None:
    if output_txt is None:
        output_txt = str(json_path.with_name(f"{json_path.stem}_speaker_analysis_research.txt"))
    if output_json is None:
        output_json = str(json_path.with_name(f"{json_path.stem}_speaker_analysis_research.json"))

    lines = []
    lines.append("FIXED DESCRIPTION")
    lines.append(interpretation["fixed_description"])
    lines.append("")
    lines.append("ADAPTIVE INTERPRETATION")
    lines.append(interpretation["adaptive_interpretation"])
    lines.append("")
    lines.append("CLASSIFICATION")
    lines.append(interpretation["classification"])
    lines.append("")
    lines.append("RESEARCH-FACING QUARTER-WISE SUMMARY")

    for q in quarter_stats:
        lines.append(
            f"{q['quarter_name'].title()}: "
            f"active_speaker_count={q['active_speaker_count']}/{len(overall['speakers'])}, "
            f"phase_share_of_meeting_time={q['phase_share_of_meeting_time'] * 100:.1f}%, "
            f"phase_share_of_total_turns={q['phase_share_of_total_turns'] * 100:.1f}%, "
            f"dominant_speaker={q['dominant_speaker']}, "
            f"dominant_share={q['dominant_share'] * 100:.1f}%, "
            f"second_speaker={q['second_speaker']}, "
            f"second_share={q['second_share'] * 100:.1f}%, "
            f"top2_concentration={q['top2_concentration'] * 100:.1f}%, "
            f"speaker_count_threshold_flag={q['speaker_count_threshold_flag']}, "
            f"dominance_strength={q['dominance_strength'] * 100:.1f}%, "
            f"turn_density={q['turn_density']:.2f} turns/min, "
            f"speaker_entry_count={q['speaker_entry_count']}"
        )

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    export_payload = {
        "fixed_description": interpretation["fixed_description"],
        "adaptive_interpretation": interpretation["adaptive_interpretation"],
        "classification": interpretation["classification"],
        "meeting_total_seconds": overall["meeting_total"],
        "meeting_total_label": sec_to_label(overall["meeting_total"]),
        "meeting_total_turns": overall["meeting_total_turns"],
        "quarters": [],
    }

    for q in quarter_stats:
        export_payload["quarters"].append(
            {
                "quarter_index": q["quarter_index"],
                "quarter_name": q["quarter_name"],
                "start_sec": round(q["start_sec"], 3),
                "end_sec": round(q["end_sec"], 3),
                "total_time_sec": round(q["total_time"], 3),
                "total_time_label": sec_to_label(q["total_time"]),
                "total_turns": q["total_turns"],
                "active_speakers": q["active_speakers"],
                "active_speaker_count": q["active_speaker_count"],
                "active_ratio": round(q["active_ratio"], 4),
                "phase_share_of_meeting_time": round(q["phase_share_of_meeting_time"], 4),
                "phase_share_of_total_turns": round(q["phase_share_of_total_turns"], 4),
                "dominant_speaker": q["dominant_speaker"],
                "dominant_share": round(q["dominant_share"], 4),
                "second_speaker": q["second_speaker"],
                "second_share": round(q["second_share"], 4),
                "top2_concentration": round(q["top2_concentration"], 4),
                "speaker_count_threshold_flag": q["speaker_count_threshold_flag"],
                "dominance_strength": round(q["dominance_strength"], 4),
                "turn_density": round(q["turn_density"], 4),
                "speaker_entry_count": q["speaker_entry_count"],
                "speaker_dominated": is_speaker_dominated(q),
            }
        )

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2, ensure_ascii=False)

    print(f"Created: {output_txt}")
    print(f"Created: {output_json}")


def build_gantt_chart(
    json_path: str,
    output_png: str | None = None,
    output_svg: str | None = None,
    output_txt: str | None = None,
    output_json: str | None = None,
) -> None:
    json_path = Path(json_path)
    df = load_meeting_json(json_path)

    if df.empty:
        raise ValueError("Input meeting JSON is empty.")

    speaker_order = sorted(df["speaker"].dropna().unique().tolist(), key=speaker_sort_key)

    base_time = datetime(2022, 1, 1, 0, 0, 0)
    df["start_dt"] = df["start"].apply(lambda s: base_time + timedelta(seconds=s))
    df["end_dt"] = df["end"].apply(lambda s: base_time + timedelta(seconds=s))
    df["width_days"] = (df["end_dt"] - df["start_dt"]).dt.total_seconds() / 86400.0

    summary = (
        df.groupby("speaker", as_index=False)
        .agg(total_speaking_time=("duration", "sum"), turns=("speaker", "count"))
    )
    summary["speaker"] = pd.Categorical(summary["speaker"], categories=speaker_order, ordered=True)
    summary = summary.sort_values("speaker").reset_index(drop=True)

    meeting_start = float(df["start"].min())
    meeting_end = float(df["end"].max())
    meeting_total = meeting_end - meeting_start
    summary["share_pct"] = summary["total_speaking_time"] / meeting_total * 100

    quarter_stats, overall = build_quarter_analysis(df)
    interpretation = classify_distribution(quarter_stats, overall)

    cmap = plt.get_cmap("tab10")
    speaker_colors = {speaker: cmap(i % 10) for i, speaker in enumerate(speaker_order)}

    fig = plt.figure(figsize=(18, 6), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[3.15, 1.28], wspace=0.16)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    ypos = {speaker: i for i, speaker in enumerate(speaker_order)}

    for _, row in df.iterrows():
        color = speaker_colors[row["speaker"]]
        ax1.barh(
            ypos[row["speaker"]],
            row["width_days"],
            left=mdates.date2num(row["start_dt"]),
            height=0.60,
            color=color,
            edgecolor=color,
            linewidth=0.0,
        )

    ax1.set_facecolor("white")
    ax1.set_yticks(list(ypos.values()))
    ax1.set_yticklabels(speaker_order, fontsize=14)
    ax1.invert_yaxis()

    tick_seconds = [meeting_total * i / 8 for i in range(9)]
    tick_positions = [mdates.date2num(base_time + timedelta(seconds=s)) for s in tick_seconds]

    ax1.set_xticks(tick_positions)
    ax1.xaxis.set_major_formatter(FuncFormatter(gantt_fmt_factory(base_time)))
    ax1.tick_params(axis="x", labelsize=13)
    ax1.tick_params(axis="y", labelsize=14)
    ax1.grid(True, axis="x", alpha=0.35, linewidth=1.0)
    ax1.grid(True, axis="y", alpha=0.15, linewidth=0.8)
    ax1.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)

    ax1.set_xlim(
        mdates.date2num(base_time),
        mdates.date2num(base_time + timedelta(seconds=meeting_end)),
    )
    ax1.set_xlabel("Meeting timeline (MM:SS)", fontsize=16)
    ax1.set_ylabel("")
    ax1.set_title("Speaking Time of Team Members in the Meeting", fontsize=26, weight="bold", pad=24)

    ax2.set_facecolor("white")
    for i, row in enumerate(summary.itertuples(index=False)):
        color = speaker_colors[str(row.speaker)]
        ax2.text(0, i - 0.34, str(row.speaker), ha="left", va="center", fontsize=12, weight="bold")
        ax2.barh(i, 100, height=0.34, color=TRACK_BLUE, edgecolor="none")
        ax2.barh(i, row.share_pct, height=0.34, color=color, edgecolor="none")
        ax2.text(
            100,
            i - 0.34,
            f"{sec_to_label(row.total_speaking_time)} | {row.share_pct:.1f}%",
            ha="right",
            va="center",
            fontsize=10.5,
        )
        ax2.text(
            50,
            i + 0.34,
            f"{int(row.turns)} Turns",
            ha="center",
            va="center",
            fontsize=10.5,
            color="#555555",
        )

    ax2.set_xlim(0, 100)
    ax2.set_ylim(-0.6, len(summary) - 0.4)
    ax2.invert_yaxis()
    ax2.set_xticks([])
    ax2.set_yticks([])

    for spine in ax2.spines.values():
        spine.set_visible(False)

    ax2.set_title("Speaker Distribution", fontsize=22, weight="bold", pad=20)
    ax2.text(
        0.5,
        1.01,
        f"Based on total meeting duration: {sec_to_label(meeting_total)}",
        transform=ax2.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
    )

    plt.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.14, wspace=0.16)

    if output_png is None:
        output_png = str(json_path.with_name(f"{json_path.stem}_gantt_research.png"))
    if output_svg is None:
        output_svg = str(json_path.with_name(f"{json_path.stem}_gantt_research.svg"))

    plt.savefig(output_png, dpi=220, bbox_inches="tight", facecolor="white")
    plt.savefig(output_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Created: {output_png}")
    print(f"Created: {output_svg}")

    save_analysis_outputs(
        json_path=json_path,
        quarter_stats=quarter_stats,
        overall=overall,
        interpretation=interpretation,
        output_txt=output_txt,
        output_json=output_json,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate meeting Gantt diagram with professor-facing and research-facing speaker analysis."
    )
    parser.add_argument("json_path", help="Path to input meeting JSON file")
    parser.add_argument("--output_png", default=None, help="Optional output PNG path")
    parser.add_argument("--output_svg", default=None, help="Optional output SVG path")
    parser.add_argument("--output_txt", default=None, help="Optional output TXT summary path")
    parser.add_argument("--output_json", default=None, help="Optional output JSON summary path")
    args = parser.parse_args()

    build_gantt_chart(
        json_path=args.json_path,
        output_png=args.output_png,
        output_svg=args.output_svg,
        output_txt=args.output_txt,
        output_json=args.output_json,
    )
