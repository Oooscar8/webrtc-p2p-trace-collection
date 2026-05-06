#!/usr/bin/env python3
"""Compute statistical significance (p-values) for QoE A/B results.

This script is designed to be used with the outputs produced by
`tools/qoe_report.py`:

- output/qoe_segments.csv: per (csv, client_id) segment metrics

It calculates statistical tests between GCC and RL groups for each scenario
and metric, and exports a tidy CSV with t-statistics / p-values.

Default choices:
- Test: Welch's t-test (two-sided)
- Unit: client-level segments (each row in qoe_segments.csv)

Notes & caveats:
- Client-level segments are not perfectly independent because two clients
  often belong to the same trace. For a more conservative test, use
  `--unit trace` to aggregate to trace-level first.
- If SciPy is not installed, Welch t-test will fall back to a permutation
  test (two-sided) using NumPy only.

Examples:
  python3 tools/qoe_significance.py --segments output/qoe_segments.csv

  # Trace-level (recommended for a more conservative conclusion)
  python3 tools/qoe_significance.py --segments output/qoe_segments.csv --unit trace

Outputs:
  - <outdir>/qoe_significance.csv
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TestResult:
    statistic: float
    dof: float | None
    p_value: float
    method: str


def welch_t_test(x: np.ndarray, y: np.ndarray, *, alternative: str) -> TestResult:
    """Welch t-test with SciPy if available.

    alternative: "two-sided" | "greater" | "less"
    """

    try:
        from scipy import stats  # type: ignore

        stat, p_two = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")

        # Compute dof (Welch–Satterthwaite) ourselves for export.
        vx = np.var(x, ddof=1)
        vy = np.var(y, ddof=1)
        nx = float(x.size)
        ny = float(y.size)
        denom = (vx / nx + vy / ny) ** 2
        dof = denom / ((vx / nx) ** 2 / (nx - 1.0) + (vy / ny) ** 2 / (ny - 1.0))

        if alternative == "two-sided":
            p = float(p_two)
        elif alternative == "greater":
            # H1: mean(x) > mean(y)
            p = float(p_two / 2.0) if stat > 0 else float(1.0 - p_two / 2.0)
        elif alternative == "less":
            # H1: mean(x) < mean(y)
            p = float(p_two / 2.0) if stat < 0 else float(1.0 - p_two / 2.0)
        else:
            raise ValueError(f"unknown alternative: {alternative}")

        return TestResult(statistic=float(stat), dof=float(dof), p_value=p, method="welch_t")
    except Exception:
        # SciPy missing or failed; fall back to permutation test.
        return permutation_test(x, y, alternative=alternative, n_resamples=20000, seed=0)


def permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alternative: str,
    n_resamples: int,
    seed: int,
) -> TestResult:
    """Permutation test for difference in means.

    Uses NumPy only. Returns a Monte-Carlo p-value.
    """

    rng = np.random.default_rng(seed)

    x = x.astype(float)
    y = y.astype(float)

    observed = float(np.mean(x) - np.mean(y))

    pooled = np.concatenate([x, y], axis=0)
    n_x = x.size

    diffs = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        rng.shuffle(pooled)
        diffs[i] = float(np.mean(pooled[:n_x]) - np.mean(pooled[n_x:]))

    if alternative == "two-sided":
        p = float((np.abs(diffs) >= abs(observed)).mean())
    elif alternative == "greater":
        p = float((diffs >= observed).mean())
    elif alternative == "less":
        p = float((diffs <= observed).mean())
    else:
        raise ValueError(f"unknown alternative: {alternative}")

    return TestResult(statistic=observed, dof=None, p_value=p, method="permutation")


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR adjustment."""

    p = np.asarray(p_values, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = np.empty_like(order)
    ranked[order] = np.arange(1, n + 1)

    q = p * n / ranked
    # Ensure monotonicity
    q_sorted = q[order]
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_adj = np.empty_like(q_sorted)
    q_adj[order] = np.clip(q_sorted, 0.0, 1.0)
    return q_adj


def validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"missing required columns in segments csv: {missing}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--segments",
        type=str,
        default="output/qoe_segments.csv",
        help="Path to output/qoe_segments.csv",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="output",
        help="Output directory (default: output)",
    )
    ap.add_argument(
        "--unit",
        choices=["client", "trace"],
        default="client",
        help="Statistical unit. 'trace' aggregates clients within a trace first.",
    )
    ap.add_argument(
        "--test",
        choices=["welch_t", "permutation"],
        default="welch_t",
        help="Statistical test. welch_t uses SciPy if available.",
    )
    ap.add_argument(
        "--alternative",
        choices=["two-sided", "greater", "less"],
        default="two-sided",
        help="Alternative hypothesis for the test.",
    )
    ap.add_argument(
        "--metrics",
        type=str,
        default="qoe_score_mean,rtt_ms_p95,loss_rate_mean,jitter_ms_p95,cap_delta_bps_mean,recv_bps_mean",
        help="Comma-separated metric columns to test.",
    )
    ap.add_argument(
        "--adjust",
        choices=["none", "bh"],
        default="bh",
        help="Multiple testing adjustment across all (scenario, metric) tests.",
    )
    ap.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="Skip tests when any group has fewer samples than this.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    seg_path = Path(args.segments)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(seg_path)

    validate_columns(df, ["scenario", "ab_group"])

    # keep only gcc/rl
    df = df[df["ab_group"].isin(["gcc", "rl"])].copy()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    for m in metrics:
        if m not in df.columns:
            raise SystemExit(f"metric column not found: {m}")

    # numeric coercion
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    # more conservative: aggregate per trace first
    if args.unit == "trace":
        validate_columns(df, ["trace_id"])
        df = (
            df.groupby(["scenario", "ab_group", "trace_id"], dropna=False)[metrics]
            .mean()
            .reset_index()
        )

    rows: list[dict] = []

    for scenario, g_s in df.groupby("scenario"):
        for metric in metrics:
            g_gcc = g_s[g_s["ab_group"] == "gcc"][metric].dropna().to_numpy(dtype=float)
            g_rl = g_s[g_s["ab_group"] == "rl"][metric].dropna().to_numpy(dtype=float)

            if g_gcc.size < args.min_samples or g_rl.size < args.min_samples:
                continue

            if args.test == "welch_t":
                tr = welch_t_test(g_rl, g_gcc, alternative=args.alternative)
            else:
                tr = permutation_test(
                    g_rl,
                    g_gcc,
                    alternative=args.alternative,
                    n_resamples=20000,
                    seed=0,
                )

            rows.append(
                {
                    "scenario": scenario,
                    "unit": args.unit,
                    "metric": metric,
                    "method": tr.method,
                    "alternative": args.alternative,
                    "n_gcc": int(g_gcc.size),
                    "n_rl": int(g_rl.size),
                    "mean_gcc": float(np.mean(g_gcc)),
                    "mean_rl": float(np.mean(g_rl)),
                    "diff_rl_minus_gcc": float(np.mean(g_rl) - np.mean(g_gcc)),
                    "statistic": float(tr.statistic),
                    "dof": float(tr.dof) if tr.dof is not None and math.isfinite(tr.dof) else "",
                    "p_value": float(tr.p_value),
                }
            )

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise SystemExit("no valid tests produced (check input / filters)")

    if args.adjust == "bh":
        out_df["p_value_bh"] = bh_fdr(out_df["p_value"].to_numpy(dtype=float))

    out_path = out_dir / "qoe_significance.csv"
    out_df.sort_values(["scenario", "metric"]).to_csv(out_path, index=False)

    print(f"written: {out_path}")
    print(
        "note: consider --unit trace for a more conservative (less dependent) test."
    )


if __name__ == "__main__":
    main()
