#!/usr/bin/env python3
"""Export a trained IQL actor checkpoint to ONNX for browser-local inference.

Example:
  python3 tools/export_iql_onnx.py \
    --model models/iql/actor.pt \
    --norm models/iql/norm.json \
    --out models/iql/actor.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.train_iql import DeterministicPolicy, GaussianPolicy  # noqa: E402


class GaussianPolicyForOnnx(nn.Module):
    def __init__(self, actor: GaussianPolicy):
        super().__init__()
        self.actor = actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out, _, _ = self.actor(obs)
        return out


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_train_config_path(model_path: Path, explicit_path: Path | None) -> Path | None:
    if explicit_path is not None:
        return explicit_path
    candidate = model_path.with_name("train_config.json")
    return candidate if candidate.exists() else None


def build_actor(model_path: Path, norm_path: Path, train_config_path: Path | None) -> tuple[nn.Module, str, int]:
    norm = load_json(norm_path)
    train_cfg = load_json(train_config_path) if train_config_path else {}

    actor_type = str(norm.get("actor_type", train_cfg.get("actor_type", "gaussian"))).lower()
    state_dim = int(norm["state_dim"])
    action_scale_bps = float(norm.get("action_scale_bps", 1_000_000.0))
    action_max_bps = float(norm.get("max_action_bps", norm.get("a_max", action_scale_bps)))
    max_action_mbps = action_max_bps / action_scale_bps
    hidden_dim = int(train_cfg.get("hidden_dim", 256))

    if actor_type == "gaussian":
        actor_dropout = train_cfg.get("actor_dropout")
        actor = GaussianPolicy(
            state_dim=state_dim,
            act_dim=1,
            max_action_mbps=max_action_mbps,
            hidden_dim=hidden_dim,
            dropout=actor_dropout,
        )
        actor.load_state_dict(torch.load(model_path, map_location="cpu"))
        wrapped: nn.Module = GaussianPolicyForOnnx(actor)
    elif actor_type == "deterministic":
        actor = DeterministicPolicy(
            state_dim=state_dim,
            act_dim=1,
            max_action_mbps=max_action_mbps,
            hidden_dim=hidden_dim,
        )
        actor.load_state_dict(torch.load(model_path, map_location="cpu"))
        wrapped = actor
    else:
        raise ValueError(f"Unsupported actor type: {actor_type}")

    wrapped.to("cpu")
    wrapped.eval()
    return wrapped, actor_type, state_dim


def export_to_onnx(model: nn.Module, actor_type: str, state_dim: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_obs = (
        torch.zeros((1, 1, state_dim), dtype=torch.float32)
        if actor_type == "gaussian"
        else torch.zeros((1, state_dim), dtype=torch.float32)
    )

    torch.onnx.export(
        model,
        (dummy_obs,),
        out_path,
        opset_version=17,
        input_names=["obs"],
        output_names=["output"],
        do_constant_folding=True,
    )


def verify_export(model: nn.Module, actor_type: str, state_dim: int, onnx_path: Path) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("Skip ONNX verification because onnxruntime is not installed.")
        return

    sample = (
        np.random.randn(1, 1, state_dim).astype(np.float32)
        if actor_type == "gaussian"
        else np.random.randn(1, state_dim).astype(np.float32)
    )
    with torch.no_grad():
        torch_out = model(torch.from_numpy(sample)).detach().cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"obs": sample})[0]
    if not np.allclose(torch_out, onnx_out, atol=1e-5):
        raise RuntimeError("PyTorch and ONNX outputs do not match closely enough.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained IQL actor to ONNX.")
    parser.add_argument("--model", type=Path, required=True, help="Path to actor.pt")
    parser.add_argument("--norm", type=Path, required=True, help="Path to norm.json")
    parser.add_argument("--out", type=Path, default=None, help="Output ONNX path (default: actor.onnx next to model)")
    parser.add_argument(
        "--train-config",
        type=Path,
        default=None,
        help="Optional train_config.json path for hidden_dim/actor_dropout metadata",
    )
    parser.add_argument("--skip-verify", action="store_true", help="Skip optional ONNXRuntime verification")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    norm_path = args.norm.resolve()
    out_path = args.out.resolve() if args.out else model_path.with_suffix(".onnx")
    train_config_path = resolve_train_config_path(model_path, args.train_config.resolve() if args.train_config else None)

    model, actor_type, state_dim = build_actor(model_path, norm_path, train_config_path)
    export_to_onnx(model, actor_type, state_dim, out_path)
    if not args.skip_verify:
        verify_export(model, actor_type, state_dim, out_path)
    print(f"Exported ONNX model to {out_path}")


if __name__ == "__main__":
    main()
