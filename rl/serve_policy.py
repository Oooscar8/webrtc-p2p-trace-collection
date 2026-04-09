#!/usr/bin/env python3
"""Serve a trained policy as HTTP/WebSocket.

Start server:
  python3 rl/serve_policy.py --model models/iql/actor.pt --norm models/iql/norm.json

HTTP:
  POST /predict

WebSocket:
  WS /ws
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel, Field


STATE_ORDER = ["send_bps", "recv_bps", "rtt_ms", "loss_rate", "jitter_ms"]


@dataclass
class Norm:
    obs_mean: np.ndarray
    obs_std: np.ndarray
    bps_scale: float
    a_min: float
    a_max: float


def transform_obs(obs: np.ndarray, norm: Norm) -> np.ndarray:
    obs = obs.astype(np.float32, copy=False)
    send_bps = np.log1p(np.maximum(obs[..., 0], 0.0) / norm.bps_scale)
    recv_bps = np.log1p(np.maximum(obs[..., 1], 0.0) / norm.bps_scale)
    rtt = np.log1p(np.maximum(obs[..., 2], 0.0))
    loss = np.clip(obs[..., 3], 0.0, 1.0)
    jitter = np.log1p(np.maximum(obs[..., 4], 0.0))

    x = np.stack([send_bps, recv_bps, rtt, loss, jitter], axis=-1)
    x = (x - norm.obs_mean) / (norm.obs_std + 1e-6)
    return x.astype(np.float32, copy=False)


def action_from_norm(a_norm: float, norm: Norm) -> float:
    a_norm = float(np.clip(a_norm, -1.0, 1.0))
    return float(norm.a_min + (a_norm + 1.0) * 0.5 * (norm.a_max - norm.a_min))


class Actor(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))


class StateIn(BaseModel):
    send_bps: float
    recv_bps: float
    rtt_ms: float
    loss_rate: float
    jitter_ms: float


class PredictIn(BaseModel):
    state: StateIn
    prev_action_bps: Optional[float] = Field(default=None)
    fallback_action_bps: Optional[float] = Field(default=None)


class PredictOut(BaseModel):
    action_bps: float
    raw_action_bps: float
    clipped: bool
    smoothed: bool
    fallback_used: bool


def load_norm(path: Path) -> Norm:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return Norm(
        obs_mean=np.asarray(cfg["obs_mean"], dtype=np.float32),
        obs_std=np.asarray(cfg["obs_std"], dtype=np.float32),
        bps_scale=float(cfg.get("bps_scale", 1e5)),
        a_min=float(cfg["a_min"]),
        a_max=float(cfg["a_max"]),
    )


def build_app(model_path: Path, norm_path: Path) -> FastAPI:
    device = torch.device("cpu")
    norm = load_norm(norm_path)

    actor = Actor(obs_dim=5)
    actor.load_state_dict(torch.load(model_path, map_location="cpu"))
    actor.eval()

    app = FastAPI()

    # smoothing state in server memory (per-process). For multi-client you should
    # move this state to the client side or key by client id.
    last_action: Optional[float] = None

    def predict_action(state: StateIn, prev_action_bps: Optional[float], fallback_action_bps: Optional[float]) -> PredictOut:
        nonlocal last_action

        obs = np.asarray(
            [state.send_bps, state.recv_bps, state.rtt_ms, state.loss_rate, state.jitter_ms],
            dtype=np.float32,
        )

        # Basic input validation & fallback
        if not np.all(np.isfinite(obs)) or state.loss_rate < 0 or state.loss_rate > 1:
            fb = float(fallback_action_bps or prev_action_bps or last_action or 200_000.0)
            last_action = fb
            return PredictOut(action_bps=fb, raw_action_bps=fb, clipped=False, smoothed=False, fallback_used=True)

        x = transform_obs(obs.reshape(1, -1), norm)
        x_t = torch.from_numpy(x).to(device)

        with torch.no_grad():
            a_norm = float(actor(x_t).cpu().numpy().reshape(()))

        raw = action_from_norm(a_norm, norm)

        # Hard clip
        clipped = False
        a = raw
        if a < norm.a_min:
            a = norm.a_min
            clipped = True
        elif a > norm.a_max:
            a = norm.a_max
            clipped = True

        # Smooth + slew-rate limit
        prev = prev_action_bps or last_action
        smoothed = False
        if prev is not None:
            prev = float(prev)
            eta_up, eta_down = 0.1, 0.3
            if a >= prev:
                a = (1 - eta_up) * prev + eta_up * a
                a = min(a, prev * 1.25)
            else:
                a = (1 - eta_down) * prev + eta_down * a
                a = max(a, prev * 0.65)
            smoothed = True

        # OOD guardrail: if action differs too much from fallback, fallback.
        if fallback_action_bps is not None:
            fb = float(fallback_action_bps)
            if fb > 0 and (a > fb * 2.0 or a < fb * 0.3):
                last_action = fb
                return PredictOut(action_bps=fb, raw_action_bps=raw, clipped=clipped, smoothed=smoothed, fallback_used=True)

        last_action = float(a)
        return PredictOut(action_bps=float(a), raw_action_bps=float(raw), clipped=clipped, smoothed=smoothed, fallback_used=False)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True, "ts": int(time.time())}

    @app.post("/predict", response_model=PredictOut)
    def predict(req: PredictIn) -> PredictOut:
        return predict_action(req.state, req.prev_action_bps, req.fallback_action_bps)

    @app.websocket("/ws")
    async def ws_endpoint(ws):
        await ws.accept()
        while True:
            msg = await ws.receive_json()
            req = PredictIn(**msg)
            out = predict_action(req.state, req.prev_action_bps, req.fallback_action_bps)
            await ws.send_json(out.dict())

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--norm", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    app = build_app(Path(args.model), Path(args.norm))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
