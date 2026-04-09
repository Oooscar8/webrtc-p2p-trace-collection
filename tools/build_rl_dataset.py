#!/usr/bin/env python3
"""Build offline-RL transitions (s, a, r, s') from WebRTC trace CSVs.

This repo collects WebRTC getStats outputs into CSV files under `real_video_csv/`.
Each CSV usually contains *both peers'* samples interleaved (distinguished by `clientId`).

This script:
- reads all CSVs under an input directory
- splits rows by `clientId`
- sorts each client's time-series by timestamp
- constructs transitions using a sliding window

Default alignment (window_size=W, stride=1):
- state_t       = aggregate(metrics[t-W+1 : t+1])
- action_t      = action_col[t]
- reward_t      = QoE(metrics[t+1], action_t, action_{t-1})
- next_state_t  = aggregate(metrics[t-W+2 : t+2])

Outputs:
- transitions.npz (D4RL-like arrays)
- transitions.csv (human readable)
- manifest.csv (per-(file,client) statistics)

Example:
  python3 tools/build_rl_dataset.py \
    --input real_video_csv \
    --output rl_dataset \
    --window-size 10
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd


DEFAULT_STATE_COLS = ["send_bps", "recv_bps", "rtt_ms", "loss_rate", "jitter_ms"]
DEFAULT_ACTION_COL = "estimated_bw_bps"

STATE_COL_ALIASES: dict[str, str] = {
    # CSV column is `jitter` (already in ms), but many papers/datasets name it `jitter_ms`.
    "jitter_ms": "jitter",
}


@dataclass(frozen=True)
class TraceMeta:
    scenario: str
    trace_id: str


_FILENAME_RE = re.compile(r"^webrtc_network_traces_(?P<scenario>.+)_(?P<trace_id>\d+)\.csv$")


def parse_trace_meta(csv_path: Path) -> TraceMeta:
    m = _FILENAME_RE.match(csv_path.name)
    if not m:
        # fallback: best-effort
        return TraceMeta(scenario="unknown", trace_id=csv_path.stem)
    return TraceMeta(scenario=m.group("scenario"), trace_id=m.group("trace_id"))


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def clean_df(df: pd.DataFrame, *, state_cols: list[str], action_col: str) -> pd.DataFrame:
    required = ["timestamp", "clientId"] + state_cols + [action_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df = df.copy()
    df = coerce_numeric(df, ["timestamp"] + state_cols + [action_col])

    # Drop rows without timestamps/clientId
    df = df.dropna(subset=["timestamp", "clientId"])

    # Basic sanitization
    for c in state_cols + [action_col]:
        # Forward-fill within each clientId to avoid mixing two peers' streams.
        df[c] = df.groupby("clientId", sort=False)[c].ffill().fillna(0)
        df[c] = df[c].astype(float)

    # Clip obviously invalid values
    if "loss_rate" in df.columns:
        df["loss_rate"] = df["loss_rate"].clip(lower=0.0, upper=1.0)
    for c in ["recv_bps", "send_bps", action_col, "rtt_ms", "jitter"]:
        if c in df.columns:
            df[c] = df[c].clip(lower=0.0)

    return df


AggMode = Literal["mean", "last", "median"]


def aggregate_window(window: pd.DataFrame, cols: list[str], agg: AggMode) -> np.ndarray:
    if agg == "last":
        row = window.iloc[-1]
        return row[cols].to_numpy(dtype=np.float32, copy=True)
    if agg == "median":
        return window[cols].median(axis=0).to_numpy(dtype=np.float32, copy=True)
    # mean
    return window[cols].mean(axis=0).to_numpy(dtype=np.float32, copy=True)


def qoe_reward(
    next_row: pd.Series,
    *,
    action: float,
    prev_action: float | None,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
) -> float:
    """A simple, bounded QoE-style reward.

    Uses next-step metrics (as a proxy for the effect after applying `action`).

    - VideoQuality: log(1 + recv_bps / 1e6)
    - DelayPenalty: rtt_ms / 1000
    - LossPenalty:  loss_rate (already 0..1)
    - Smoothness:   |a_t - a_{t-1}| / 1e6
    """

    recv_bps = float(next_row.get("recv_bps", 0.0) or 0.0)
    rtt_ms = float(next_row.get("rtt_ms", 0.0) or 0.0)
    loss_rate = float(next_row.get("loss_rate", 0.0) or 0.0)

    video_quality = math.log1p(max(recv_bps, 0.0) / 1_000_000.0)
    delay_penalty = max(rtt_ms, 0.0) / 1000.0
    loss_penalty = min(max(loss_rate, 0.0), 1.0)

    if prev_action is None:
        smoothness = 0.0
    else:
        smoothness = abs(float(action) - float(prev_action)) / 1_000_000.0

    return float(alpha * video_quality - beta * delay_penalty - gamma * loss_penalty - delta * smoothness)


@dataclass
class TransitionBatch:
    observations: list[np.ndarray]
    actions: list[float]
    rewards: list[float]
    next_observations: list[np.ndarray]
    terminals: list[int]
    metadf_rows: list[dict]


def build_transitions_for_client(
    df_client: pd.DataFrame,
    *,
    meta: TraceMeta,
    csv_path: Path,
    client_id: str,
    state_cols: list[str],
    action_col: str,
    window_size: int,
    stride: int,
    agg: AggMode,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    min_action: float | None,
) -> TransitionBatch:
    df_client = df_client.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    n = len(df_client)
    if n < window_size + 1:
        return TransitionBatch([], [], [], [], [], [])

    observations: list[np.ndarray] = []
    actions: list[float] = []
    rewards: list[float] = []
    next_observations: list[np.ndarray] = []
    terminals: list[int] = []
    metadf_rows: list[dict] = []

    # t is the index of action/current step; reward uses t+1; next_state ends at t+1
    for t in range(window_size - 1, n - 1, stride):
        action = float(df_client.at[t, action_col])
        if min_action is not None and action < min_action:
            continue

        state_win = df_client.iloc[t - window_size + 1 : t + 1]
        next_state_win = df_client.iloc[t - window_size + 2 : t + 2]

        obs = aggregate_window(state_win, state_cols, agg)
        next_obs = aggregate_window(next_state_win, state_cols, agg)

        prev_action = float(df_client.at[t - 1, action_col]) if t - 1 >= 0 else None
        reward = qoe_reward(
            df_client.iloc[t + 1],
            action=action,
            prev_action=prev_action,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
        )

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        next_observations.append(next_obs)
        terminals.append(1 if t + 1 >= n - 1 else 0)

        metadf_rows.append(
            {
                "csv": str(csv_path),
                "scenario": meta.scenario,
                "trace_id": meta.trace_id,
                "client_id": client_id,
                "t": int(t),
                "timestamp": int(df_client.at[t, "timestamp"]),
            }
        )

    return TransitionBatch(observations, actions, rewards, next_observations, terminals, metadf_rows)


def load_csv(csv_path: Path, *, state_cols: list[str], action_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return clean_df(df, state_cols=state_cols, action_col=action_col)


def list_csv_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("*.csv"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="real_video_csv", help="CSV file or directory")
    parser.add_argument("--output", type=str, default="rl_dataset", help="Output directory")

    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--agg", choices=["mean", "last", "median"], default="mean")

    parser.add_argument(
        "--state-cols",
        type=str,
        default=",".join(DEFAULT_STATE_COLS),
        help="Comma-separated state columns, in order",
    )
    parser.add_argument("--action-col", type=str, default=DEFAULT_ACTION_COL)

    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.1)

    parser.add_argument(
        "--min-action",
        type=float,
        default=1.0,
        help="Drop samples with action < min-action. Use 0 to keep zero-action rows.",
    )
    parser.add_argument(
        "--format",
        choices=["npz", "csv", "both"],
        default="both",
        help="Output format",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    window_size = int(args.window_size)
    stride = int(args.stride)
    if window_size < 2:
        raise SystemExit("--window-size must be >= 2")
    if stride < 1:
        raise SystemExit("--stride must be >= 1")

    state_cols_out = [c.strip() for c in str(args.state_cols).split(",") if c.strip()]
    if not state_cols_out:
        raise SystemExit("--state-cols is empty")

    state_cols_in = [STATE_COL_ALIASES.get(c, c) for c in state_cols_out]

    action_col = str(args.action_col)

    csv_files = list_csv_files(input_path)
    if not csv_files:
        raise SystemExit(f"no csv files found under: {input_path}")

    all_obs: list[np.ndarray] = []
    all_actions: list[float] = []
    all_rewards: list[float] = []
    all_next_obs: list[np.ndarray] = []
    all_terminals: list[int] = []
    all_meta_rows: list[dict] = []

    manifest_rows: list[dict] = []

    for csv_path in csv_files:
        meta = parse_trace_meta(csv_path)
        df = load_csv(csv_path, state_cols=state_cols_in, action_col=action_col)

        for client_id, df_client in df.groupby("clientId"):
            batch = build_transitions_for_client(
                df_client,
                meta=meta,
                csv_path=csv_path,
                client_id=str(client_id),
                state_cols=state_cols_in,
                action_col=action_col,
                window_size=window_size,
                stride=stride,
                agg=args.agg,
                alpha=float(args.alpha),
                beta=float(args.beta),
                gamma=float(args.gamma),
                delta=float(args.delta),
                min_action=None if float(args.min_action) <= 0 else float(args.min_action),
            )

            manifest_rows.append(
                {
                    "csv": str(csv_path),
                    "scenario": meta.scenario,
                    "trace_id": meta.trace_id,
                    "client_id": str(client_id),
                    "rows": int(len(df_client)),
                    "transitions": int(len(batch.actions)),
                }
            )

            all_obs.extend(batch.observations)
            all_actions.extend(batch.actions)
            all_rewards.extend(batch.rewards)
            all_next_obs.extend(batch.next_observations)
            all_terminals.extend(batch.terminals)
            all_meta_rows.extend(batch.metadf_rows)

    if not all_actions:
        raise SystemExit("no transitions generated (check --min-action / --window-size)")

    observations = np.stack(all_obs).astype(np.float32)
    actions = np.asarray(all_actions, dtype=np.float32)
    rewards = np.asarray(all_rewards, dtype=np.float32)
    next_observations = np.stack(all_next_obs).astype(np.float32)
    terminals = np.asarray(all_terminals, dtype=np.int8)

    meta_df = pd.DataFrame(all_meta_rows)
    manifest_df = pd.DataFrame(manifest_rows)

    # Deterministic ordering for csv readability
    if not meta_df.empty:
        meta_df = meta_df.reset_index(drop=True)

    # Save
    manifest_df.to_csv(output_dir / "manifest.csv", index=False)

    if args.format in ("csv", "both"):
        data: dict[str, np.ndarray] = {}
        for i, col in enumerate(state_cols_out):
            data[f"s_{col}"] = observations[:, i]
        data["a_estimated_bw_bps"] = actions
        data["r"] = rewards
        for i, col in enumerate(state_cols_out):
            data[f"s1_{col}"] = next_observations[:, i]
        data["done"] = terminals

        transitions_df = pd.DataFrame(data)
        transitions_df = pd.concat([meta_df, transitions_df], axis=1)
        transitions_df.to_csv(output_dir / "transitions.csv", index=False)

    if args.format in ("npz", "both"):
        np.savez_compressed(
            output_dir / "transitions.npz",
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            terminals=terminals,
            state_cols=np.asarray(state_cols_out, dtype=object),
            state_cols_source=np.asarray(state_cols_in, dtype=object),
            action_col=np.asarray([action_col], dtype=object),
        )

    print(
        "generated transitions:",
        {
            "count": int(actions.shape[0]),
            "obs_shape": list(observations.shape),
            "actions_shape": list(actions.shape),
            "output": str(output_dir),
        },
    )


if __name__ == "__main__":
    main()
