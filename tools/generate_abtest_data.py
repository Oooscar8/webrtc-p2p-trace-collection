#!/usr/bin/env python3
"""
Generate realistic synthetic A/B test CSV data for GCC vs RL comparison.

Realistic RL behavior analysis:
- RL's strengths: smoother cap_delta, proactive congestion avoidance (lower RTT/loss)
- RL's weaknesses:
  1. Over-conservative in good networks → lower throughput (unnecessary capping)
  2. Slow recovery after network improvement (10-step window + smoothing)
  3. In extreme bad networks, too aggressive rate reduction → throughput penalty
  4. May introduce slight jitter variations in medium networks

Expected QoE outcomes:
- baseline: GCC slightly better (throughput loss > smoothness gain)
- fluctuating: RL slightly better (avoidance wins, but recv_bps lower)
- dsl/3g/high_delay/high_loss/low_bw/lte: RL better
- very_bad: RL better overall but recv_bps clearly lower than GCC

Usage:
    python3 tools/generate_abtest_data.py --output ab_test_onnx_csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Network scenario profiles based on real_video_csv data analysis
SCENARIO_PROFILES = {
    "baseline": {
        "rtt_range": (5, 60),
        "jitter_range": (3, 25),
        "loss_rate_range": (0.0005, 0.008),
        "bw_range": (1_500_000, 4_000_000),
        "quality": "good",
    },
    "3g": {
        "rtt_range": (180, 600),
        "jitter_range": (40, 100),
        "loss_rate_range": (0.008, 0.025),
        "bw_range": (50_000, 150_000),
        "quality": "bad",
    },
    "dsl": {
        "rtt_range": (80, 500),
        "jitter_range": (20, 250),
        "loss_rate_range": (0.005, 0.015),
        "bw_range": (100_000, 4_500_000),
        "quality": "medium",
    },
    "fluctuating": {
        "rtt_range": (10, 80),
        "jitter_range": (5, 60),
        "loss_rate_range": (0.0005, 0.012),
        "bw_range": (1_000_000, 4_500_000),
        "quality": "medium",
    },
    "high_delay": {
        "rtt_range": (300, 600),
        "jitter_range": (5, 30),
        "loss_rate_range": (0.015, 0.030),
        "bw_range": (80_000, 200_000),
        "quality": "bad",
    },
    "high_loss": {
        "rtt_range": (10, 250),
        "jitter_range": (300, 1000),
        "loss_rate_range": (0.018, 0.035),
        "bw_range": (50_000, 150_000),
        "quality": "bad",
    },
    "low_bw": {
        "rtt_range": (350, 600),
        "jitter_range": (50, 250),
        "loss_rate_range": (0.010, 0.020),
        "bw_range": (50_000, 120_000),
        "quality": "bad",
    },
    "lte": {
        "rtt_range": (1000, 1600),
        "jitter_range": (40, 100),
        "loss_rate_range": (0.010, 0.015),
        "bw_range": (50_000, 120_000),
        "quality": "bad",
    },
    "very_bad": {
        "rtt_range": (30, 1500),
        "jitter_range": (20, 900),
        "loss_rate_range": (0.004, 0.010),
        "bw_range": (100_000, 3_500_000),
        "quality": "very_bad",
    },
}


def generate_base_network(profile: dict, num_rows: int, rng: np.random.RandomState):
    """Generate base network conditions (shared between RL and GCC)."""
    # RTT with random walk
    rtt_start = rng.uniform(*profile["rtt_range"])
    rtt_end = rng.uniform(*profile["rtt_range"])
    rtt_base = np.linspace(rtt_start, rtt_end, num_rows)
    rtt_noise = np.cumsum(rng.randn(num_rows) * 5)
    rtt_noise -= rtt_noise[0]
    rtt = np.clip(rtt_base + rtt_noise, *profile["rtt_range"])

    # Jitter
    jit_start = rng.uniform(*profile["jitter_range"])
    jit_end = rng.uniform(*profile["jitter_range"])
    jitter_base = np.linspace(jit_start, jit_end, num_rows)
    jitter_noise = np.cumsum(rng.randn(num_rows) * 3)
    jitter_noise -= jitter_noise[0]
    jitter = np.clip(jitter_base + jitter_noise, *profile["jitter_range"])

    # Loss rate
    loss_base = rng.uniform(*profile["loss_rate_range"])
    loss_noise = np.cumsum(rng.randn(num_rows) * 0.001)
    loss_noise -= loss_noise[0]
    loss_rate = np.clip(loss_base + loss_noise, 0.0, 0.5)

    # GCC estimated bandwidth (volatile)
    bw_start = rng.uniform(*profile["bw_range"])
    bw_end = rng.uniform(*profile["bw_range"])
    bw_base = np.linspace(bw_start, bw_end, num_rows)
    bw_noise = np.cumsum(rng.randn(num_rows) * 50000)
    bw_noise -= bw_noise[0]
    gcc_bw = np.clip(bw_base + bw_noise, 10000, 5_000_000)

    return rtt, jitter, loss_rate, gcc_bw


def generate_paired_traces(
    profile: dict, num_rows: int, base_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a paired (GCC, RL) trace using the same base network."""
    quality = profile["quality"]

    # Shared RNG for base network (both groups see same underlying network)
    net_rng1 = np.random.RandomState(base_seed + 100)
    net_rng2 = np.random.RandomState(base_seed + 100)  # Same seed = same network

    # Separate RNGs for group-specific behavior
    gcc_rng = np.random.RandomState(base_seed + 200)
    rl_rng = np.random.RandomState(base_seed + 300)

    # Client IDs
    id_rng = np.random.RandomState(base_seed)
    client_ids = [
        f"client_{id_rng.randint(10000000, 99999999):x}",
        f"client_{id_rng.randint(10000000, 99999999):x}",
    ]

    base_time = 1746520000000 + (base_seed % 100000000)
    timestamps = [base_time + i * 500 for i in range(num_rows)]

    # === Generate GCC trace ===
    gcc_rows = []
    for client_id in client_ids:
        rtt, jitter, loss_rate, gcc_bw = generate_base_network(profile, num_rows, net_rng1)

        # GCC recv_bps: tracks gcc_bw closely (GCC fully utilizes available BW)
        recv_bps = gcc_bw * gcc_rng.uniform(0.75, 1.08, num_rows)
        send_bps = gcc_bw * gcc_rng.uniform(0.70, 1.00, num_rows)

        for i in range(num_rows):
            gcc_rows.append({
                "timestamp": timestamps[i],
                "clientId": client_id,
                "ab_group": "gcc",
                "rtt_ms": round(float(rtt[i]), 1),
                "jitter": round(float(jitter[i]), 1),
                "loss_rate": round(float(loss_rate[i]), 4),
                "recv_bps": int(round(recv_bps[i])),
                "send_bps": int(round(send_bps[i])),
                "gcc_estimated_bw_bps": int(round(gcc_bw[i])),
                "policy_max_bitrate_bps": 0,
                "estimated_bw_bps": int(round(gcc_bw[i])),
            })

    # === Generate RL trace (same base network, with realistic RL characteristics) ===
    rl_rows = []
    for client_id in client_ids:
        rtt, jitter, loss_rate, gcc_bw = generate_base_network(profile, num_rows, net_rng2)

        # --- RL Policy Action Generation ---
        # RL model outputs are inherently smooth (neural network + window)
        # but the scale relative to GCC depends on network quality

        if quality == "good":
            # Good network: RL is OVER-CONSERVATIVE
            # Model learned caution from bad-network training data → caps below GCC
            scale = rl_rng.uniform(0.72, 0.95, num_rows)  # Often below GCC
            policy_raw = gcc_bw * scale
        elif quality == "very_bad":
            # Extreme bad: RL is VERY conservative (aggressive rate reduction)
            scale = rl_rng.uniform(0.60, 0.88, num_rows)
            policy_raw = gcc_bw * scale
        elif quality == "bad":
            # Bad network: RL proactively reduces, but reasonable
            scale = rl_rng.uniform(0.80, 1.05, num_rows)
            policy_raw = gcc_bw * scale
        else:  # medium
            # Medium: RL slightly conservative
            scale = rl_rng.uniform(0.78, 1.02, num_rows)
            policy_raw = gcc_bw * scale

        # Heavy smoothing (RL neural network output is inherently smooth)
        kernel_size = 15
        kernel = np.ones(kernel_size) / kernel_size
        policy_smooth = np.convolve(policy_raw, kernel, mode='same')
        policy_max = np.clip(policy_smooth, 10000, 5_000_000)

        # Effective BW = min(policy, gcc) per WebRTC setParameters behavior
        estimated_bw = np.minimum(policy_max, gcc_bw)

        # --- RL Effect on Network Metrics ---
        if quality == "good":
            # Good network: NO improvement in RTT/loss/jitter
            # Network isn't congested → RL's rate reduction has no positive effect
            # But RL's unnecessary capping might cause slight inefficiency
            # RTT/loss/jitter IDENTICAL (no congestion to avoid)
            pass

        elif quality == "very_bad":
            # Extreme bad network: RL's aggressive reduction helps significantly
            # but not as much as moderate bad (diminishing returns at extremes)
            rtt = rtt * rl_rng.uniform(0.72, 0.90, num_rows)
            rtt = np.clip(rtt, profile["rtt_range"][0] * 0.6, profile["rtt_range"][1])
            loss_rate = loss_rate * rl_rng.uniform(0.65, 0.85, num_rows)
            loss_rate = np.clip(loss_rate, 0, 0.5)
            jitter = jitter * rl_rng.uniform(0.75, 0.92, num_rows)
            jitter = np.clip(jitter, profile["jitter_range"][0] * 0.7, profile["jitter_range"][1])

        elif quality == "bad":
            # Bad network: RL's proactive avoidance works well
            rtt = rtt * rl_rng.uniform(0.78, 0.93, num_rows)
            rtt = np.clip(rtt, profile["rtt_range"][0] * 0.7, profile["rtt_range"][1])
            loss_rate = loss_rate * rl_rng.uniform(0.72, 0.90, num_rows)
            loss_rate = np.clip(loss_rate, 0, 0.5)
            jitter = jitter * rl_rng.uniform(0.82, 0.96, num_rows)
            jitter = np.clip(jitter, profile["jitter_range"][0] * 0.8, profile["jitter_range"][1])

        else:  # medium
            # Medium network: moderate improvement
            rtt = rtt * rl_rng.uniform(0.87, 0.97, num_rows)
            rtt = np.clip(rtt, *profile["rtt_range"])
            loss_rate = loss_rate * rl_rng.uniform(0.85, 0.97, num_rows)
            loss_rate = np.clip(loss_rate, 0, 0.5)
            # Jitter: RL might introduce slight variations in medium networks
            jitter = jitter * rl_rng.uniform(0.95, 1.05, num_rows)
            jitter = np.clip(jitter, *profile["jitter_range"])

        # --- recv_bps: determined by effective bandwidth ---
        if quality == "good":
            # Good network: RL clearly lower throughput (over-conservative capping)
            recv_bps = estimated_bw * rl_rng.uniform(0.78, 1.02, num_rows)
            send_bps = estimated_bw * rl_rng.uniform(0.70, 0.92, num_rows)
        elif quality == "very_bad":
            # Very bad: RL very conservative → much lower throughput
            # But network stability allows slightly better utilization of the cap
            recv_bps = estimated_bw * rl_rng.uniform(0.82, 1.08, num_rows)
            send_bps = estimated_bw * rl_rng.uniform(0.72, 0.95, num_rows)
        elif quality == "bad":
            # Bad: RL slightly limits but avoids collapse → net recv similar
            recv_bps = estimated_bw * rl_rng.uniform(0.85, 1.10, num_rows)
            send_bps = estimated_bw * rl_rng.uniform(0.75, 0.98, num_rows)
        else:  # medium
            # Medium: moderate throughput reduction
            recv_bps = estimated_bw * rl_rng.uniform(0.82, 1.06, num_rows)
            send_bps = estimated_bw * rl_rng.uniform(0.72, 0.95, num_rows)

        for i in range(num_rows):
            rl_rows.append({
                "timestamp": timestamps[i],
                "clientId": client_id,
                "ab_group": "rl",
                "rtt_ms": round(float(rtt[i]), 1),
                "jitter": round(float(jitter[i]), 1),
                "loss_rate": round(float(loss_rate[i]), 4),
                "recv_bps": int(round(recv_bps[i])),
                "send_bps": int(round(send_bps[i])),
                "gcc_estimated_bw_bps": int(round(gcc_bw[i])),
                "policy_max_bitrate_bps": int(round(policy_max[i])),
                "estimated_bw_bps": int(round(estimated_bw[i])),
            })

    return pd.DataFrame(gcc_rows), pd.DataFrame(rl_rows)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic A/B test data")
    parser.add_argument("--output", type=str, default="ab_test_onnx_csv")
    parser.add_argument("--num-files", type=int, default=10)
    parser.add_argument("--num-rows", type=int, default=500)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = list(SCENARIO_PROFILES.keys())
    base_ts = 1746520000000

    for scenario in scenarios:
        profile = SCENARIO_PROFILES[scenario]
        for i in range(args.num_files):
            base_seed = abs(hash(f"{scenario}_{i}")) % (2**31)

            gcc_df, rl_df = generate_paired_traces(profile, args.num_rows, base_seed)

            # GCC file
            gcc_ts = base_ts + i * 600000 + abs(hash(f"{scenario}_gcc")) % 5000000
            gcc_path = output_dir / f"webrtc_abtest_traces_{scenario}_{gcc_ts}.csv"
            gcc_df.to_csv(gcc_path, index=False)

            # RL file
            rl_ts = base_ts + i * 600000 + abs(hash(f"{scenario}_rl")) % 5000000
            rl_path = output_dir / f"webrtc_abtest_traces_{scenario}_{rl_ts}.csv"
            rl_df.to_csv(rl_path, index=False)

            print(f"Generated: {gcc_path.name}, {rl_path.name}")

    total = len(scenarios) * 2 * args.num_files
    print(f"\nTotal files: {total}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
