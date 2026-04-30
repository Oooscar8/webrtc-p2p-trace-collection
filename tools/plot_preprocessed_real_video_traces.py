#!/usr/bin/env python3
"""Plot raw vs repaired WebRTC trace sessions.

Usage:
  python3 tools/plot_preprocessed_real_video_traces.py \
      --input real_video_csv \
      --outdir output/real_video_session_plots_compare

The script recursively scans CSV files under the input directory, applies the
same preprocessing pipeline used by RL dataset building, and emits one PNG per
CSV with 5 rows x 2 columns:
  left  = raw trace
  right = repaired trace after:
    - invalid zero-point detection
    - short-gap interpolation
    - long-gap splitting

Output subdirectories mirror the relative layout under --input, so scenario
folders remain separated when the dataset is organized by network condition.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps

from build_rl_dataset import DEFAULT_ACTION_COL, DEFAULT_STATE_COLS, clean_df


RAW_COLUMNS = [
    "timestamp",
    "clientId",
    "rtt_ms",
    "jitter",
    "loss_rate",
    "recv_bps",
    "send_bps",
    "estimated_bw_bps",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot raw vs repaired RTT/jitter/loss/throughput/GCC trends for every CSV session."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("real_video_csv"),
        help="Directory containing trace CSV files, optionally in nested subdirectories.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("output/real_video_session_plots_compare"),
        help="Directory for generated PNG figures.",
    )
    parser.add_argument("--dpi", type=int, default=150, help="PNG DPI.")
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=1,
        help="Optional rolling mean window size for smoothing. Use 1 to disable.",
    )
    parser.add_argument(
        "--max-short-gap",
        type=int,
        default=3,
        help="Linearly repair NaN/invalid gaps up to this many consecutive samples.",
    )
    return parser.parse_args()


def ensure_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "jitter_ms" in df.columns and "jitter" not in df.columns:
        df = df.rename(columns={"jitter_ms": "jitter"})

    missing = [col for col in RAW_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    return df


def coerce_plot_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["timestamp", "rtt_ms", "jitter", "loss_rate", "recv_bps", "estimated_bw_bps"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "clientId"])
    df["clientId"] = df["clientId"].astype(str)
    return df


def build_plot_columns(df: pd.DataFrame, *, base_ts: float, rolling_window: int) -> pd.DataFrame:
    sort_cols = ["clientId", "timestamp"]
    if "segment_id" in df.columns:
        sort_cols = ["clientId", "segment_id", "timestamp"]
    df = df.sort_values(sort_cols, kind="mergesort").copy()
    df["elapsed_s"] = (df["timestamp"] - base_ts) / 1000.0
    df["loss_pct"] = df["loss_rate"].clip(lower=0.0, upper=1.0) * 100.0
    df["recv_mbps"] = df["recv_bps"].clip(lower=0.0) / 1_000_000.0
    df["gcc_mbps"] = df["estimated_bw_bps"].clip(lower=0.0) / 1_000_000.0

    if rolling_window > 1:
        value_cols = ["rtt_ms", "jitter", "loss_pct", "recv_mbps", "gcc_mbps"]
        group_cols = ["clientId"] + (["segment_id"] if "segment_id" in df.columns else [])
        for col in value_cols:
            df[col] = df.groupby(group_cols, sort=False)[col].transform(
                lambda s: s.rolling(window=rolling_window, min_periods=1).mean()
            )

    return df


def build_client_colors(client_ids: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = colormaps["tab10"]
    return {client_id: cmap(idx % cmap.N) for idx, client_id in enumerate(client_ids)}


def plot_df_on_axes(
    df: pd.DataFrame,
    axes: list[plt.Axes],
    *,
    title: str,
    segmented: bool,
    color_by_client: dict[str, tuple[float, float, float, float]],
) -> None:
    metrics = [
        ("rtt_ms", "RTT (ms)"),
        ("jitter", "Jitter (ms)"),
        ("loss_pct", "Loss Rate (%)"),
        ("recv_mbps", "Throughput recv_bps (Mbps)"),
        ("gcc_mbps", "GCC Estimated BW (Mbps)"),
    ]

    group_keys = ["clientId", "segment_id"] if segmented else ["clientId"]
    handles = []
    labels = []

    for ax, (column, ylabel) in zip(axes, metrics):
        for key, sub_df in df.groupby(group_keys, sort=False):
            client_id = str(key[0] if isinstance(key, tuple) else key)
            line = ax.plot(
                sub_df["elapsed_s"],
                sub_df[column],
                linewidth=1.5,
                label=client_id,
                color=color_by_client[client_id],
            )[0]
            if ylabel == "RTT (ms)" and client_id not in labels:
                handles.append(line)
                labels.append(client_id)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    axes[0].set_title(title)
    axes[-1].set_xlabel("Elapsed Time (s)")

    if handles:
        axes[0].legend(handles, labels, loc="upper right", fontsize=8)


def plot_session(csv_path: Path, out_path: Path, *, max_short_gap: int, rolling_window: int, dpi: int) -> None:
    raw_df = pd.read_csv(csv_path)
    raw_df = ensure_raw_columns(raw_df)
    raw_df = coerce_plot_columns(raw_df)
    if raw_df.empty:
        raise ValueError("raw dataframe is empty after parsing")

    base_ts = float(raw_df["timestamp"].min())
    raw_plot_df = build_plot_columns(raw_df, base_ts=base_ts, rolling_window=rolling_window)

    repaired_df = clean_df(
        raw_df,
        state_cols=[col if col != "jitter_ms" else "jitter" for col in DEFAULT_STATE_COLS],
        action_col=DEFAULT_ACTION_COL,
        max_short_gap=max_short_gap,
    )
    if repaired_df.empty:
        raise ValueError("repaired dataframe is empty")

    repaired_df["clientId"] = repaired_df["clientId"].astype(str)
    repaired_plot_df = build_plot_columns(repaired_df, base_ts=base_ts, rolling_window=rolling_window)

    client_ids = list(
        pd.Index(raw_plot_df["clientId"].astype(str))
        .union(pd.Index(repaired_plot_df["clientId"].astype(str)))
        .unique()
    )
    color_by_client = build_client_colors(client_ids)

    fig, axes = plt.subplots(5, 2, figsize=(18, 18), sharex="col")
    left_axes = list(axes[:, 0])
    right_axes = list(axes[:, 1])

    raw_title = f"Raw\nclients={raw_plot_df['clientId'].nunique()} rows={len(raw_plot_df)}"
    repaired_title = (
        "Repaired\n"
        f"clients={repaired_plot_df['clientId'].nunique()} rows={len(repaired_plot_df)} "
        f"segments={repaired_plot_df['segment_id'].nunique()} short_gap<={max_short_gap}"
    )

    plot_df_on_axes(
        raw_plot_df,
        left_axes,
        title=raw_title,
        segmented=False,
        color_by_client=color_by_client,
    )
    plot_df_on_axes(
        repaired_plot_df,
        right_axes,
        title=repaired_title,
        segmented=True,
        color_by_client=color_by_client,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(f"{csv_path.name}\nsmooth_window={rolling_window}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = args.input.expanduser().resolve()
    out_dir = args.outdir.expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input path is not a directory: {input_dir}")

    csv_files = sorted(input_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"no csv files found in: {input_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failed: list[tuple[str, str]] = []

    for csv_path in csv_files:
        relative_parent = csv_path.relative_to(input_dir).parent
        out_path = out_dir / relative_parent / f"{csv_path.stem}_compare.png"
        try:
            plot_session(
                csv_path,
                out_path,
                max_short_gap=max(0, int(args.max_short_gap)),
                rolling_window=max(1, int(args.rolling_window)),
                dpi=int(args.dpi),
            )
            success_count += 1
            print(f"[ok] {csv_path} -> {out_path}")
        except Exception as exc:  # noqa: BLE001
            failed.append((str(csv_path), str(exc)))
            print(f"[skip] {csv_path}: {exc}")

    print(f"\nDone. Generated {success_count} comparison figure(s) in {out_dir}")
    if failed:
        print("Failed files:")
        for name, reason in failed:
            print(f"  - {name}: {reason}")


if __name__ == "__main__":
    main()
