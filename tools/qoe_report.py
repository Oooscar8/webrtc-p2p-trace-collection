#!/usr/bin/env python3
"""Generate QoE metrics report for GCC vs RL (A/B).

This script reads CSV traces collected by this repo and computes:
- Per-segment QoE metrics (grouped by csv file + clientId)
- Aggregated metrics by (scenario, ab_group)

Usage:
  python3 tools/qoe_report.py --input real_video_csv --outdir output

Outputs:
  - <outdir>/qoe_segments.csv
  - <outdir>/qoe_ab_summary.csv

Notes:
- It is backward compatible with old CSV headers.
- QoE score here is a simple, reproducible proxy suitable for A/B comparison:
    qoe = log(1 + recv_bps / 1e6)
          - rtt_ms / 1000
          - loss_rate
          - 0.1 * |cap_t - cap_{t-1}| / 1e6
  where cap_t is the effective sending cap:
    - RL group: policy_max_bitrate_bps (if available)
    - GCC group: gcc_estimated_bw_bps (if available)
    - otherwise: estimated_bw_bps
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


FILENAME_RE = re.compile(
    r"^(?:webrtc_network_traces|webrtc_abtest_traces)_(?P<scenario>.+?)_(?P<trace_id>\d+)\.csv$"
)


def parse_filename(p: Path) -> tuple[str, str]:
    m = FILENAME_RE.match(p.name)
    if not m:
        return "unknown", p.stem
    return m.group("scenario"), m.group("trace_id")


def ensure_col(df: pd.DataFrame, name: str, default: float = 0.0) -> None:
    if name not in df.columns:
        df[name] = default


def qoe_for_group(df: pd.DataFrame, *, ab_group: str) -> dict:
    if df.empty:
        return {}

    # numeric coercion
    for c in [
        "timestamp",
        "rtt_ms",
        "jitter_ms",
        "loss_rate",
        "recv_bps",
        "send_bps",
        "gcc_estimated_bw_bps",
        "policy_max_bitrate_bps",
        "estimated_bw_bps",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    duration_s = 0.0
    if df["timestamp"].max() > 0 and df["timestamp"].min() > 0:
        duration_s = max(0.0, (df["timestamp"].max() - df["timestamp"].min()) / 1000.0)

    if ab_group == "rl":
        cap = df["policy_max_bitrate_bps"].where(df["policy_max_bitrate_bps"] > 0, df["estimated_bw_bps"])
    else:
        cap = df["gcc_estimated_bw_bps"].where(df["gcc_estimated_bw_bps"] > 0, df["estimated_bw_bps"])

    cap = cap.clip(lower=0.0)
    cap_delta = cap.diff().abs().fillna(0.0)

    qoe = (
        np.log1p(df["recv_bps"] / 1e6)
        - (df["rtt_ms"] / 1000.0)
        - df["loss_rate"].clip(0.0, 1.0)
        - 0.1 * (cap_delta / 1e6)
    )

    return {
        "duration_s": float(duration_s),
        "recv_bps_mean": float(df["recv_bps"].mean()),
        "send_bps_mean": float(df["send_bps"].mean()),
        "rtt_ms_p95": float(df["rtt_ms"].quantile(0.95)),
        "jitter_ms_p95": float(df["jitter_ms"].quantile(0.95)),
        "loss_rate_mean": float(df["loss_rate"].mean()),
        "cap_bps_mean": float(cap.mean()),
        "cap_delta_bps_mean": float(cap_delta.mean()),
        "qoe_score_mean": float(qoe.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="real_video_csv")
    ap.add_argument("--outdir", type=str, default="output")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for csv_path in sorted(in_dir.glob("*.csv")):
        scenario, trace_id = parse_filename(csv_path)

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] failed to read {csv_path}: {e}")
            continue

        # backward compatible renames
        if "jitter" in df.columns and "jitter_ms" not in df.columns:
            df = df.rename(columns={"jitter": "jitter_ms"})

        # expected columns (old CSV may miss these)
        ensure_col(df, "ab_group", "unknown")
        ensure_col(df, "gcc_estimated_bw_bps", 0.0)
        ensure_col(df, "policy_max_bitrate_bps", 0.0)
        ensure_col(df, "estimated_bw_bps", 0.0)

        # some very old CSVs might not have these exact names
        for c in ["timestamp", "clientId", "rtt_ms", "loss_rate", "recv_bps", "send_bps", "jitter_ms"]:
            ensure_col(df, c, 0.0)

        for client_id, g in df.groupby("clientId"):
            if str(client_id) == "0" and g["timestamp"].sum() == 0:
                continue

            ab = str(g["ab_group"].iloc[0] if "ab_group" in g.columns else "unknown")
            ab = "rl" if ab == "rl" else ("gcc" if ab == "gcc" else "unknown")

            metrics = qoe_for_group(g.copy(), ab_group=ab)
            if not metrics:
                continue

            rows.append(
                {
                    "csv": csv_path.name,
                    "scenario": scenario,
                    "trace_id": trace_id,
                    "client_id": str(client_id),
                    "ab_group": ab,
                    **metrics,
                }
            )

    seg_df = pd.DataFrame(rows)
    seg_out = out_dir / "qoe_segments.csv"
    seg_df.to_csv(seg_out, index=False)

    if seg_df.empty:
        print(f"written: {seg_out} (empty)")
        return

    # scenario-level A/B aggregation
    agg = (
        seg_df.groupby(["scenario", "ab_group"], dropna=False)
        .agg(
            traces=("trace_id", "nunique"),
            clients=("client_id", "nunique"),
            duration_s_mean=("duration_s", "mean"),
            recv_bps_mean=("recv_bps_mean", "mean"),
            rtt_ms_p95_mean=("rtt_ms_p95", "mean"),
            loss_rate_mean=("loss_rate_mean", "mean"),
            jitter_ms_p95_mean=("jitter_ms_p95", "mean"),
            cap_bps_mean=("cap_bps_mean", "mean"),
            cap_delta_bps_mean=("cap_delta_bps_mean", "mean"),
            qoe_score_mean=("qoe_score_mean", "mean"),
        )
        .reset_index()
        .sort_values(["scenario", "ab_group"])
    )

    agg_out = out_dir / "qoe_ab_summary.csv"
    agg.to_csv(agg_out, index=False)

    print(f"written: {seg_out}")
    print(f"written: {agg_out}")


if __name__ == "__main__":
    main()