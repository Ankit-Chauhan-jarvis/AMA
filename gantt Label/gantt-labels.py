import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from matplotlib.ticker import FuncFormatter

ORDER = [
    "Problem-oriented statements",
    "Procedural statements",
    "Socio-emotional statements",
    "Action-oriented statements",
    "Counterproductive/Destructive statements",
]

LABEL_COLORS = {
    "Problem-oriented statements": "blue",
    "Procedural statements": "orange",
    "Socio-emotional statements": "red",
    "Action-oriented statements": "purple",
    "Counterproductive/Destructive statements": "green",
}

LABEL_MAP = {
    "problem": "Problem-oriented statements",
    "problem-oriented": "Problem-oriented statements",
    "problem oriented": "Problem-oriented statements",
    "problem-oriented statements": "Problem-oriented statements",
    "problem statements": "Problem-oriented statements",
    "task": "Problem-oriented statements",
    "procedural": "Procedural statements",
    "procedural statements": "Procedural statements",
    "social": "Socio-emotional statements",
    "socioemotional": "Socio-emotional statements",
    "socio-emotional": "Socio-emotional statements",
    "socio emotional": "Socio-emotional statements",
    "socio-emotional statements": "Socio-emotional statements",
    "socioemotional statements": "Socio-emotional statements",
    "action": "Action-oriented statements",
    "action-oriented": "Action-oriented statements",
    "action oriented": "Action-oriented statements",
    "action-oriented statements": "Action-oriented statements",
    "counter": "Counterproductive/Destructive statements",
    "counterproductive": "Counterproductive/Destructive statements",
    "destructive": "Counterproductive/Destructive statements",
    "counterproductive/destructive": "Counterproductive/Destructive statements",
    "counterproductive / destructive": "Counterproductive/Destructive statements",
    "counterproductive-destructive": "Counterproductive/Destructive statements",
    "counterproductive/destructive statements": "Counterproductive/Destructive statements",
    "counterproductive / destructive statements": "Counterproductive/Destructive statements",
    "destructive statements": "Counterproductive/Destructive statements",
    "problemorientierte aussagen": "Problem-oriented statements",
    "problemorientierte aussagen (+)": "Problem-oriented statements",
    "prozedurale aussagen": "Procedural statements",
    "prozedurale aussagen (+)": "Procedural statements",
    "sozioemotionale aussagen": "Socio-emotional statements",
    "sozioemotionale aussagen (+)": "Socio-emotional statements",
    "maßnahmenorientierte aussagen": "Action-oriented statements",
    "maßnahmenorientierte aussagen (+)": "Action-oriented statements",
    "massnahmenorientierte aussagen": "Action-oriented statements",
    "massnahmenorientierte aussagen (+)": "Action-oriented statements",
    "kontraproduktive/destruktive aussagen": "Counterproductive/Destructive statements",
    "kontraproduktive / destruktive aussagen": "Counterproductive/Destructive statements",
}

REQUIRED_COLUMN_GROUPS = {
    "start_time": ["start_time", "start", "start time"],
    "end_time": ["end_time", "end", "end time"],
    "label": ["label", "labels", "act4teams", "code", "category"],
}

BAR_HEIGHT = 0.55
TIME_SEGMENTS = 8


def sec_to_mmss(value, _):
    total_seconds = max(0, int(round(value)))
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None


def normalize_label(value) -> str | None:
    if pd.isna(value):
        return None
    key = str(value).strip().lower()
    return LABEL_MAP.get(key) if key else None


def load_meeting_data(input_file: Path) -> pd.DataFrame:
    df = pd.read_excel(input_file)

    resolved = {}
    for target, candidates in REQUIRED_COLUMN_GROUPS.items():
        column = find_column(df, candidates)
        if column is None:
            raise ValueError(
                f"Missing required column for '{target}' in {input_file.name}. "
                f"Available columns: {list(df.columns)}"
            )
        resolved[target] = column

    work = df[[resolved["start_time"], resolved["end_time"], resolved["label"]]].copy()
    work.columns = ["start_time", "end_time", "label"]
    work["start_time"] = pd.to_numeric(work["start_time"], errors="coerce")
    work["end_time"] = pd.to_numeric(work["end_time"], errors="coerce")
    work["label_final"] = work["label"].apply(normalize_label)
    work = work.dropna(subset=["start_time", "end_time"]).copy()
    work = work[work["end_time"] > work["start_time"]].copy()
    work["duration"] = work["end_time"] - work["start_time"]

    if work.empty:
        raise ValueError(f"No valid time rows found in {input_file.name}.")

    return work.sort_values(["start_time", "end_time"]).reset_index(drop=True)


def create_xticks(meeting_end: float, segments: int = TIME_SEGMENTS) -> list[float]:
    return [meeting_end * i / segments for i in range(segments + 1)]


def draw_label_segments(ax, subset: pd.DataFrame, row_index: int, color: str):
    if subset.empty:
        return
    segments = list(zip(subset["start_time"].tolist(), subset["duration"].tolist()))
    ax.broken_barh(
        segments,
        (row_index - BAR_HEIGHT / 2, BAR_HEIGHT),
        facecolors=color,
        edgecolors=color,
        linewidth=0.6,
    )


def draw_side_panel(ax_side, label_stats: dict, meeting_end: float):
    ax_side.set_facecolor("white")
    for spine in ax_side.spines.values():
        spine.set_visible(False)
    ax_side.set_xticks([])
    ax_side.set_yticks([])

    total_m, total_s = divmod(int(meeting_end), 60)
    ax_side.text(
        0, 1.06, "Label Distribution",
        transform=ax_side.transAxes,
        fontsize=13, fontweight="bold", va="bottom", ha="left",
    )
    ax_side.text(
        0, 1.01,
        f"Based on total meeting duration: {total_m}m {total_s:02d}s",
        transform=ax_side.transAxes,
        fontsize=8.5, color="gray", va="bottom", ha="left",
    )

    n = len(ORDER)
    max_dur = max(s["duration"] for s in label_stats.values()) or 1

    slot_h = 1.0
    bar_h = 0.28
    header_off = 0.78
    bar_center = 0.50
    turns_off = 0.20

    ax_side.set_xlim(0, 1)
    ax_side.set_ylim(0, n)

    for i, label_name in enumerate(ORDER):
        stats = label_stats[label_name]
        
        slot_bottom = (n - 1 - i) * slot_h
        color = LABEL_COLORS[label_name]
        bar_fill = stats["duration"] / max_dur

        dur_m, dur_s = divmod(int(stats["duration"]), 60)
        dur_str = f"{dur_m}m {dur_s:02d}s"
        pct_str = f"{stats['pct']:.1f}%"

        
        y_header = slot_bottom + header_off * slot_h
        ax_side.text(0, y_header, label_name,
                     fontsize=9, fontweight="bold", color="black",
                     va="center", ha="left")
        ax_side.text(1.0, y_header, f"{dur_str} | {pct_str}",
                     fontsize=8.5, color="black",
                     va="center", ha="right")

        
        y_bar = slot_bottom + bar_center * slot_h
        ax_side.barh(y_bar, 1.0, height=bar_h * slot_h,
                     left=0, color="#dce6f5", zorder=1)
        if bar_fill > 0:
            ax_side.barh(y_bar, bar_fill, height=bar_h * slot_h,
                         left=0, color=color, zorder=2)

        
        y_turns = slot_bottom + turns_off * slot_h
        ax_side.text(0, y_turns, f"{stats['turns']} Turns",
                     fontsize=8, color="gray",
                     va="center", ha="left")


def build_gantt(input_file: Path, output_file: Path):
    work = load_meeting_data(input_file)
    labeled = work.dropna(subset=["label_final"]).copy()
    meeting_end = float(work["end_time"].max())
    total_duration = float(work["duration"].sum())

    label_stats = {}
    for label_name in ORDER:
        subset = labeled[labeled["label_final"] == label_name]
        dur = float(subset["duration"].sum())
        label_stats[label_name] = {
            "duration": dur,
            "turns": len(subset),
            "pct": dur / total_duration * 100 if total_duration > 0 else 0,
        }

    fig = plt.figure(figsize=(24, 7), facecolor="white")
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax_side = fig.add_subplot(gs[1])

    
    ax.set_facecolor("white")
    for row_index, label_name in enumerate(ORDER):
        subset = labeled[labeled["label_final"] == label_name]
        draw_label_segments(ax, subset, row_index, LABEL_COLORS[label_name])

    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels(ORDER, fontsize=12)
    ax.invert_yaxis()
    ax.set_ylim(len(ORDER) - 0.5, -0.5)
    ax.set_xticks(create_xticks(meeting_end))
    ax.xaxis.set_major_formatter(FuncFormatter(sec_to_mmss))
    ax.set_xlabel("Meeting time (mm:ss)", fontsize=12)
    ax.set_title("Meeting Content by Function", fontsize=24, fontweight="bold", pad=16)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    
    draw_side_panel(ax_side, label_stats, meeting_end)

    
    fig.add_artist(plt.Line2D(
        [ax_side.get_position().x0 - 0.005, ax_side.get_position().x0 - 0.005],
        [0.08, 0.92],
        transform=fig.transFigure,
        color="#cccccc", linewidth=1,
    ))

    plt.savefig(output_file, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def collect_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    files = sorted([*input_path.glob("*.xls"), *input_path.glob("*.xlsx")])
    if not files:
        raise FileNotFoundError(f"No .xls or .xlsx files found in: {input_path}")
    return files


def main():
    parser = argparse.ArgumentParser(description="Create Gantt diagrams from labeled meeting Excel files.")
    parser.add_argument("--input", required=True, help="Excel file or folder containing Excel files.")
    parser.add_argument("--output_dir", default="gantt_output", help="Folder to save generated PNG files.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_file in collect_input_files(input_path):
        output_file = output_dir / f"{input_file.stem}_gantt.png"
        try:
            build_gantt(input_file, output_file)
            print(f"Saved: {output_file}")
        except Exception as exc:
            print(f"Failed: {input_file.name} -> {exc}")


if __name__ == "__main__":
    main()