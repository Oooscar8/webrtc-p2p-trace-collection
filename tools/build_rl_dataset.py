#!/usr/bin/env python3
"""Build offline-RL transitions (s, a, r, s') from WebRTC trace CSVs.

Compared with the previous version, this builder keeps the full 10-step history
instead of aggregating the window into a single 5-d vector. Before building
transitions, it also applies a lightweight trace-cleaning pipeline tailored for
real WebRTC CSVs:

- identify invalid periodic zero points (for example, RTT/BWE drops to 0 while
  traffic is still flowing)
- linearly repair only short gaps
- split long gaps into separate valid segments so transitions never cross them

- state_t      = 10 rows of [send_bps, recv_bps, rtt_ms, loss_rate, jitter_ms]
- action_t     = estimated_bw_bps at the current row
- reward_t     = unchanged QoE-style reward
- next_state_t = the next 10-row window

The exported state is flattened in feature-major order to match the historical
window layout used by Schaferct-style training:

  [send_bps(t-9:t), recv_bps(t-9:t), rtt_ms(t-9:t), loss_rate(t-9:t), jitter_ms(t-9:t)]

To align the normalization operation with Schaferct, state features are scaled
with fixed per-feature coefficients and then clipped to a bounded range.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_STATE_COLS = ["send_bps", "recv_bps", "rtt_ms", "loss_rate", "jitter_ms"]
DEFAULT_ACTION_COL = "estimated_bw_bps"

STATE_COL_ALIASES: dict[str, str] = {
    "jitter_ms": "jitter",
}

STATE_LAYOUT = "feature_major"
STATE_SCALE_BY_COL: dict[str, float] = {
    "send_bps": 1e-6,
    "recv_bps": 1e-6,
    "rtt_ms": 1e-2,
    "loss_rate": 1.0,
    "jitter_ms": 1e-2,
}
NORMALIZATION_METHOD = "schaferct_fixed_scale_clip"
NORMALIZATION_CLIP_RANGE = (-10.0, 10.0)


@dataclass(frozen=True)
class TraceMeta:
    scenario: str
    trace_id: str


@dataclass
class TransitionBatch:
    observations: list[np.ndarray]
    actions: list[float]
    rewards: list[float]
    next_observations: list[np.ndarray]
    terminals: list[int]
    metadf_rows: list[dict]


_FILENAME_RE = re.compile(r"^webrtc_network_traces_(?P<scenario>.+)_(?P<trace_id>\d+)\.csv$")


def parse_trace_meta(csv_path: Path) -> TraceMeta:
    match = _FILENAME_RE.match(csv_path.name)
    if not match:
        return TraceMeta(scenario="unknown", trace_id=csv_path.stem)
    return TraceMeta(scenario=match.group("scenario"), trace_id=match.group("trace_id"))


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def mark_invalid_zero_points(df: pd.DataFrame, *, action_col: str) -> pd.DataFrame:
    df = df.copy()

    def neighbor_positive(series: pd.Series) -> pd.Series:
        return series.shift(1).fillna(0).gt(0) | series.shift(-1).fillna(0).gt(0)

    recv_positive = (
        df["recv_bps"].fillna(0).gt(0) if "recv_bps" in df.columns else pd.Series(False, index=df.index)
    )
    send_positive = (
        df["send_bps"].fillna(0).gt(0) if "send_bps" in df.columns else pd.Series(False, index=df.index)
    )
    loss_positive = (
        df["loss_rate"].fillna(0).gt(0) if "loss_rate" in df.columns else pd.Series(False, index=df.index)
    )
    jitter_positive = (
        df["jitter"].fillna(0).gt(0) if "jitter" in df.columns else pd.Series(False, index=df.index)
    )
    activity = recv_positive | send_positive | loss_positive | jitter_positive

    if "rtt_ms" in df.columns:
        rtt = df["rtt_ms"].fillna(0)
        invalid_rtt = rtt.le(0) & (activity | neighbor_positive(rtt))
        df.loc[invalid_rtt, "rtt_ms"] = np.nan

    if action_col in df.columns:
        action = df[action_col].fillna(0)
        invalid_action = action.le(0) & (activity | neighbor_positive(action))
        df.loc[invalid_action, action_col] = np.nan

    return df


def interpolate_short_gaps_series(series: pd.Series, *, max_gap: int) -> pd.Series:
    if max_gap <= 0 or not series.isna().any():
        return series

    is_na = series.isna()
    groups = is_na.ne(is_na.shift()).cumsum()
    gap_sizes = is_na.groupby(groups).transform("sum")
    short_gap_mask = is_na & gap_sizes.le(max_gap)
    interpolated = series.interpolate(method="linear", limit_area="inside")

    filled = series.copy()
    filled.loc[short_gap_mask] = interpolated.loc[short_gap_mask]
    return filled


def repair_short_gaps(
    df: pd.DataFrame,
    *,
    cols: list[str],
    max_gap: int,
) -> pd.DataFrame:
    if max_gap <= 0:
        return df

    df = df.copy()
    for col in cols:
        df[col] = df.groupby("clientId", sort=False)[col].transform(
            lambda s: interpolate_short_gaps_series(s.astype(float), max_gap=max_gap)
        )
    return df


def assign_segment_ids(df: pd.DataFrame, *, required_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    segment_ids = pd.Series(-1, index=df.index, dtype=np.int64)

    for _, client_df in df.groupby("clientId", sort=False):
        valid = client_df[required_cols].notna().all(axis=1)
        starts = valid & ~valid.shift(fill_value=False)
        ids = starts.cumsum() - 1
        ids = ids.where(valid, -1).astype(np.int64)
        segment_ids.loc[client_df.index] = ids

    df["segment_id"] = segment_ids
    return df


def clean_df(
    df: pd.DataFrame,
    *,
    state_cols: list[str],
    action_col: str,
    max_short_gap: int,
) -> pd.DataFrame:
    """在构建 transition 之前，对单个原始 trace DataFrame 做预处理。

    预处理遵循三个原则：
    1. 先保留能够确定时序和身份的最小必要字段；
    2. 把可疑的周期性零点当作缺失值，而不是真实网络测量值，再只修复短 gap；
    3. 对长 gap 进行切段，保证后续状态窗口不会跨越断裂区域。

    返回值仍然是逐行的时序数据；后续阶段才会基于这些清洗后的行去构建
    state window 和 transition。
    """
    # 先检查原始 CSV 是否包含训练所需的状态列和动作列。
    # 如果缺少这些字段，后续就无法判断一行是否有效，也无法安全构建样本。
    required = ["timestamp", "clientId"] + state_cols + [action_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df = df.copy()
    # 先把相关列统一转成数值；无法解析的字符串会被转成 NaN，
    # 这样后续的缺失修复和切段逻辑就能统一处理。
    df = coerce_numeric(df, ["timestamp"] + state_cols + [action_col])

    # 没有 timestamp 或 clientId 的行无法参与按客户端排序的时序轨迹，
    # 因此直接丢弃。
    df = df.dropna(subset=["timestamp", "clientId"])

    # 在插值前，先按 clientId 和 timestamp 排序。
    # 这里使用稳定排序，尽量保留相同时间戳行的原始相对顺序。
    df = df.sort_values(["clientId", "timestamp"], kind="mergesort")

    # 如果同一个 client 出现重复时间戳，只保留最后一条，
    # 保证每个时刻最多对应一行观测。
    df = df.drop_duplicates(subset=["clientId", "timestamp"], keep="last")

    # 在识别无效零点之前，先做一次最基础的物理范围裁剪：
    # throughput / bandwidth / RTT / jitter 不应该出现负值。
    for col in ["recv_bps", "send_bps", action_col, "rtt_ms", "jitter"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0.0)

    # loss_rate 应该位于 [0, 1] 区间。
    # 这一步不是在修复缺失，只是在约束明显越界的数值。
    if "loss_rate" in df.columns:
        df["loss_rate"] = df["loss_rate"].clip(lower=0.0, upper=1.0)

    repaired_clients: list[pd.DataFrame] = []
    # 按 client 分别处理。对于 RTT=0 / 估计带宽=0 这种点，如果周围样本
    # 仍然显示有流量或相邻观测是非零值，就更像是异常采样点而不是真实网络状态，
    # 因此先重新标记为 NaN。
    for _, client_df in df.groupby("clientId", sort=False):
        repaired_clients.append(mark_invalid_zero_points(client_df, action_col=action_col))
    df = pd.concat(repaired_clients, axis=0).sort_index()

    # 这里只对短缺失段做线性插值修复。
    # 这样既能修补短暂采样空洞，又避免跨长断点“脑补”出不可信的轨迹。
    df = repair_short_gaps(df, cols=state_cols + [action_col], max_gap=max_short_gap)

    # 经过短 gap 修复后，如果仍然存在 NaN，说明这段缺失太长
    # 或该行本身仍然无效。此时只给 state 和 action 都完整的连续区域
    # 分配 segment_id，用断点把轨迹切开。
    df = assign_segment_ids(df, required_cols=state_cols + [action_col])

    # 最后只保留属于有效连续 segment 的行。
    # 后续构建 transition 时会按 (clientId, segment_id) 分组，因此窗口不会跨越长断点。
    df = df[df["segment_id"] >= 0].copy()

    return df


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


def window_to_feature_major(window: pd.DataFrame, cols: list[str]) -> np.ndarray:
    values = window[cols].to_numpy(dtype=np.float32, copy=True)
    return values.T.copy()


def build_state_scales(state_cols: list[str]) -> np.ndarray:
    missing = [col for col in state_cols if col not in STATE_SCALE_BY_COL]
    if missing:
        raise ValueError(f"Missing Schaferct-style fixed scales for columns: {missing}")
    return np.asarray([STATE_SCALE_BY_COL[col] for col in state_cols], dtype=np.float32)


def normalize_feature_windows(
    windows: np.ndarray,
    *,
    feature_scales: np.ndarray,
    clip_range: tuple[float, float],
) -> np.ndarray:
    normalized = windows * feature_scales[None, :, None]
    normalized = np.clip(normalized, clip_range[0], clip_range[1])
    return normalized.astype(np.float32, copy=False)


def flatten_feature_major_windows(windows: np.ndarray) -> np.ndarray:
    return windows.reshape(windows.shape[0], -1).astype(np.float32, copy=False)


def build_column_names(prefix: str, state_cols: list[str], window_size: int) -> list[str]:
    names: list[str] = []
    for col in state_cols:
        for lag in range(window_size - 1, -1, -1):
            names.append(f"{prefix}_{col}_t-{lag}")
    return names


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
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    min_action: float | None,
) -> TransitionBatch:
    df_client = df_client.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    n_rows = len(df_client)
    if n_rows < window_size + 1:
        return TransitionBatch([], [], [], [], [], [])

    observations: list[np.ndarray] = []
    actions: list[float] = []
    rewards: list[float] = []
    next_observations: list[np.ndarray] = []
    terminals: list[int] = []
    metadf_rows: list[dict] = []

    for t in range(window_size - 1, n_rows - 1, stride):
        action = float(df_client.at[t, action_col])
        if min_action is not None and action < min_action:
            continue

        state_window = df_client.iloc[t - window_size + 1 : t + 1]
        next_state_window = df_client.iloc[t - window_size + 2 : t + 2]

        obs = window_to_feature_major(state_window, state_cols)
        next_obs = window_to_feature_major(next_state_window, state_cols)

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
        terminals.append(1 if t + 1 >= n_rows - 1 else 0)
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


def load_csv(
    csv_path: Path,
    *,
    state_cols: list[str],
    action_col: str,
    max_short_gap: int,
) -> pd.DataFrame:
    return clean_df(
        pd.read_csv(csv_path),
        state_cols=state_cols,
        action_col=action_col,
        max_short_gap=max_short_gap,
    )


def list_csv_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.rglob("*.csv"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="real_video_csv", help="CSV file or directory")
    parser.add_argument("--output", type=str, default="rl_dataset", help="Output directory")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--agg",
        choices=["mean", "last", "median"],
        default="last",
        help="Deprecated compatibility flag. Raw 10-step windows are always used.",
    )
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
        "--max-short-gap",
        type=int,
        default=3,
        help="Linearly repair NaN/invalid gaps up to this many consecutive samples; longer gaps split the trace.",
    )
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

    state_cols_out = [col.strip() for col in str(args.state_cols).split(",") if col.strip()]
    if not state_cols_out:
        raise SystemExit("--state-cols is empty")
    state_cols_in = [STATE_COL_ALIASES.get(col, col) for col in state_cols_out]
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
        df = load_csv(
            csv_path,
            state_cols=state_cols_in,
            action_col=action_col,
            max_short_gap=int(args.max_short_gap),
        )

        for (client_id, segment_id), df_client in df.groupby(["clientId", "segment_id"]):
            batch = build_transitions_for_client(
                df_client,
                meta=meta,
                csv_path=csv_path,
                client_id=str(client_id),
                state_cols=state_cols_in,
                action_col=action_col,
                window_size=window_size,
                stride=stride,
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
                    "segment_id": int(segment_id),
                    "rows": int(len(df_client)),
                    "transitions": int(len(batch.actions)),
                }
            )

            for row in batch.metadf_rows:
                row["segment_id"] = int(segment_id)

            all_obs.extend(batch.observations)
            all_actions.extend(batch.actions)
            all_rewards.extend(batch.rewards)
            all_next_obs.extend(batch.next_observations)
            all_terminals.extend(batch.terminals)
            all_meta_rows.extend(batch.metadf_rows)

    if not all_actions:
        raise SystemExit("no transitions generated (check --min-action / --window-size)")

    observations_raw = np.stack(all_obs).astype(np.float32)
    next_observations_raw = np.stack(all_next_obs).astype(np.float32)
    actions = np.asarray(all_actions, dtype=np.float32)
    rewards = np.asarray(all_rewards, dtype=np.float32)
    terminals = np.asarray(all_terminals, dtype=np.int8)

    feature_scales = build_state_scales(state_cols_out)

    observations = flatten_feature_major_windows(
        normalize_feature_windows(
            observations_raw,
            feature_scales=feature_scales,
            clip_range=NORMALIZATION_CLIP_RANGE,
        )
    )
    next_observations = flatten_feature_major_windows(
        normalize_feature_windows(
            next_observations_raw,
            feature_scales=feature_scales,
            clip_range=NORMALIZATION_CLIP_RANGE,
        )
    )

    meta_df = pd.DataFrame(all_meta_rows).reset_index(drop=True)
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(output_dir / "manifest.csv", index=False)

    norm_payload = {
        "state_cols": state_cols_out,
        "state_cols_source": state_cols_in,
        "window_size": window_size,
        "state_dim": int(observations.shape[1]),
        "state_layout": STATE_LAYOUT,
        "normalization_method": NORMALIZATION_METHOD,
        "state_scales": feature_scales.tolist(),
        "clip_min": NORMALIZATION_CLIP_RANGE[0],
        "clip_max": NORMALIZATION_CLIP_RANGE[1],
        "action_col": action_col,
        "observations_normalized": True,
    }
    (output_dir / "norm.json").write_text(json.dumps(norm_payload, indent=2), encoding="utf-8")

    if args.format in ("csv", "both"):
        data: dict[str, np.ndarray] = {}
        state_columns = build_column_names("s", state_cols_out, window_size)
        next_state_columns = build_column_names("s1", state_cols_out, window_size)
        for index, column_name in enumerate(state_columns):
            data[column_name] = observations[:, index]
        data["a_estimated_bw_bps"] = actions
        data["r"] = rewards
        for index, column_name in enumerate(next_state_columns):
            data[column_name] = next_observations[:, index]
        data["done"] = terminals
        transitions_df = pd.concat([meta_df, pd.DataFrame(data)], axis=1)
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
            normalization_method=np.asarray([NORMALIZATION_METHOD], dtype=object),
            state_scales=feature_scales,
            clip_min=np.asarray([NORMALIZATION_CLIP_RANGE[0]], dtype=np.float32),
            clip_max=np.asarray([NORMALIZATION_CLIP_RANGE[1]], dtype=np.float32),
            window_size=np.asarray([window_size], dtype=np.int32),
            state_dim=np.asarray([observations.shape[1]], dtype=np.int32),
            state_layout=np.asarray([STATE_LAYOUT], dtype=object),
            observations_normalized=np.asarray([True], dtype=bool),
        )

    print(
        "generated transitions:",
        {
            "count": int(actions.shape[0]),
            "obs_shape": list(observations.shape),
            "actions_shape": list(actions.shape),
            "window_size": window_size,
            "state_layout": STATE_LAYOUT,
            "output": str(output_dir),
        },
    )


if __name__ == "__main__":
    main()
