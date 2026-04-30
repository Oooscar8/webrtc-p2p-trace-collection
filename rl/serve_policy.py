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
from collections import deque
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


STATE_ORDER = ["send_bps", "recv_bps", "rtt_ms", "loss_rate", "jitter_ms"]
STATE_SCALE_BY_COL = {
    "send_bps": 1e-6,
    "recv_bps": 1e-6,
    "rtt_ms": 1e-2,
    "loss_rate": 1.0,
    "jitter_ms": 1e-2,
}
NORMALIZATION_CLIP_RANGE = (-10.0, 10.0)


@dataclass
class Norm:
    state_cols: list[str]
    window_size: int
    state_dim: int
    state_layout: str
    normalization_method: str
    feature_mean: np.ndarray
    feature_std: np.ndarray
    state_scales: np.ndarray
    clip_min: float
    clip_max: float
    observations_normalized: bool
    action_scale_bps: float
    action_min_bps: float
    action_max_bps: float
    actor_type: str


@dataclass
class InferenceState:
    history: Deque[np.ndarray]
    last_action_bps: Optional[float] = None


def build_state_scales(state_cols: list[str]) -> np.ndarray:
    missing = [col for col in state_cols if col not in STATE_SCALE_BY_COL]
    if missing:
        raise ValueError(f"Missing fixed scales for columns: {missing}")
    return np.asarray([STATE_SCALE_BY_COL[col] for col in state_cols], dtype=np.float32)


def standardize_window(window: np.ndarray, norm: Norm) -> np.ndarray:
    feature_mean = np.asarray(norm.feature_mean, dtype=np.float32)
    feature_std = np.asarray(norm.feature_std, dtype=np.float32)
    if feature_mean.size == window.shape[0]:
        return (window - feature_mean[:, None]) / (feature_std[:, None] + 1e-6)
    if feature_mean.size == window.size:
        flat = window.reshape(-1)
        flat = (flat - feature_mean) / (feature_std + 1e-6)
        return flat.reshape(window.shape)
    raise ValueError(
        f"Unsupported normalization shape: mean={feature_mean.size} std={feature_std.size} window={window.shape}"
    )


def transform_window(window: np.ndarray, norm: Norm) -> np.ndarray:
    window = window.astype(np.float32, copy=False)
    if norm.normalization_method == "legacy_log1p_standardize":
        send_bps = np.log1p(np.maximum(window[0], 0.0) / 1e5)
        recv_bps = np.log1p(np.maximum(window[1], 0.0) / 1e5)
        rtt = np.log1p(np.maximum(window[2], 0.0))
        loss = np.clip(window[3], 0.0, 1.0)
        jitter = np.log1p(np.maximum(window[4], 0.0))
        window = np.stack([send_bps, recv_bps, rtt, loss, jitter], axis=0)
        window = standardize_window(window, norm)
    else:
        window = window * norm.state_scales[:, None]
        window = np.clip(window, norm.clip_min, norm.clip_max)
        if norm.feature_mean.size > 0 and norm.feature_std.size > 0:
            window = standardize_window(window, norm)

    flat = window.reshape(-1).astype(np.float32, copy=False)
    if flat.size != norm.state_dim:
        raise ValueError(f"state_dim mismatch: expected {norm.state_dim}, got {flat.size}")
    return flat


def build_windowed_state(history: Deque[np.ndarray], norm: Norm) -> np.ndarray:
    if not history:
        raise ValueError("Empty history")
    states = list(history)
    if len(states) < norm.window_size:
        states = [states[0]] * (norm.window_size - len(states)) + states
    else:
        states = states[-norm.window_size :]
    window = np.stack(states, axis=0).astype(np.float32, copy=False)
    return window.T.copy()


class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, max_action_mbps: float, hidden: int = 256):
        super().__init__()
        self.max_action_mbps = float(max_action_mbps)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden, hidden, 2)
        self.fc_mid = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.rb1 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.Identity(),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
        )
        self.rb2 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.Identity(),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
        )
        self.final = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs
        add_sequence_dim = False
        if x.ndim == 3 and x.shape[0] == 1:
            x = x.squeeze(0)
            add_sequence_dim = True
        x = self.encoder(x)
        x, _ = self.gru(x)
        x = self.fc_mid(x)
        residual = x
        x = self.rb1(x) + residual
        residual = x
        x = self.rb2(x) + residual
        mean = (self.final(x) * self.max_action_mbps).clamp(min=1e-5)
        std = torch.exp(self.log_std.clamp(-20.0, 2.0)).view(1, -1).expand(mean.shape[0], -1)
        out = torch.cat((mean, std), dim=-1)
        if add_sequence_dim:
            out = out.unsqueeze(0)
        return out


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim: int, max_action_mbps: float, hidden: int = 256):
        super().__init__()
        self.max_action_mbps = float(max_action_mbps)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.net(obs) * self.max_action_mbps, min=1e-5)


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
    state_cols = [str(col) for col in cfg.get("state_cols", STATE_ORDER)]
    window_size = int(cfg.get("window_size", 1))
    state_dim = int(cfg.get("state_dim", len(state_cols) * window_size))
    state_layout = str(cfg.get("state_layout", "feature_major"))
    normalization_method = str(
        cfg.get(
            "normalization_method",
            "legacy_log1p_standardize" if "a_min" in cfg or "a_max" in cfg else "schaferct_fixed_scale_clip",
        )
    )
    feature_mean = np.asarray(cfg.get("feature_mean", cfg.get("obs_mean", [])), dtype=np.float32)
    feature_std = np.asarray(cfg.get("feature_std", cfg.get("obs_std", [])), dtype=np.float32)
    state_scales = np.asarray(cfg.get("state_scales", build_state_scales(state_cols)), dtype=np.float32)
    clip_min = float(cfg.get("clip_min", NORMALIZATION_CLIP_RANGE[0]))
    clip_max = float(cfg.get("clip_max", NORMALIZATION_CLIP_RANGE[1]))
    action_scale_bps = float(cfg.get("action_scale_bps", 1_000_000.0))
    action_min_bps = float(cfg.get("a_min", 1e-5 * action_scale_bps))
    action_max_bps = float(cfg.get("a_max", cfg.get("max_action_bps", float("inf"))))
    return Norm(
        state_cols=state_cols,
        window_size=window_size,
        state_dim=state_dim,
        state_layout=state_layout,
        normalization_method=normalization_method,
        feature_mean=feature_mean,
        feature_std=feature_std,
        state_scales=state_scales,
        clip_min=clip_min,
        clip_max=clip_max,
        observations_normalized=bool(cfg.get("observations_normalized", False)),
        action_scale_bps=action_scale_bps,
        action_min_bps=action_min_bps,
        action_max_bps=action_max_bps,
        actor_type=str(cfg.get("actor_type", "deterministic")),
    )


def load_actor(model_path: Path, norm: Norm) -> nn.Module:
    if norm.state_layout != "feature_major":
        raise ValueError(f"Unsupported state layout: {norm.state_layout}")

    max_action_mbps = norm.action_max_bps / norm.action_scale_bps
    if norm.actor_type == "gaussian":
        actor: nn.Module = GaussianActor(obs_dim=norm.state_dim, max_action_mbps=max_action_mbps)
    else:
        actor = DeterministicActor(obs_dim=norm.state_dim, max_action_mbps=max_action_mbps)
    actor.load_state_dict(torch.load(model_path, map_location="cpu"))
    actor.eval()
    return actor


def infer_raw_action_bps(actor: nn.Module, x: np.ndarray, norm: Norm, device: torch.device) -> float:
    x_t = torch.from_numpy(x.astype(np.float32, copy=False)).to(device)
    with torch.no_grad():
        if norm.actor_type == "gaussian":
            out = actor(x_t.reshape(1, 1, -1))
            raw_action_mbps = float(out[..., 0].cpu().numpy().reshape(-1)[0])
        else:
            out = actor(x_t.reshape(1, -1))
            raw_action_mbps = float(out.cpu().numpy().reshape(-1)[0])
    return raw_action_mbps * norm.action_scale_bps


def build_app(model_path: Path, norm_path: Path) -> FastAPI:
    device = torch.device("cpu")
    norm = load_norm(norm_path)
    actor = load_actor(model_path, norm)

    app = FastAPI()

    # Add CORS middleware to allow cross-origin requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # HTTP uses a shared process-level state window. WebSocket connections each get
    # an isolated window to avoid mixing histories across clients.
    http_state = InferenceState(history=deque(maxlen=norm.window_size))

    def predict_action(
        state: StateIn,
        prev_action_bps: Optional[float],
        fallback_action_bps: Optional[float],
        inference_state: InferenceState,
    ) -> PredictOut:
        obs = np.asarray(
            [state.send_bps, state.recv_bps, state.rtt_ms, state.loss_rate, state.jitter_ms],
            dtype=np.float32,
        )

        # Basic input validation & fallback
        if not np.all(np.isfinite(obs)) or state.loss_rate < 0 or state.loss_rate > 1:
            fb = float(fallback_action_bps or prev_action_bps or inference_state.last_action_bps or 200_000.0)
            inference_state.last_action_bps = fb
            return PredictOut(action_bps=fb, raw_action_bps=fb, clipped=False, smoothed=False, fallback_used=True)

        inference_state.history.append(obs)
        window = build_windowed_state(inference_state.history, norm)
        x = transform_window(window, norm)
        raw = infer_raw_action_bps(actor, x, norm, device)

        # Hard clip
        clipped = False
        a = raw
        if a < norm.action_min_bps:
            a = norm.action_min_bps
            clipped = True
        elif a > norm.action_max_bps:
            a = norm.action_max_bps
            clipped = True

        # Smooth + slew-rate limit
        prev = prev_action_bps or inference_state.last_action_bps
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
                inference_state.last_action_bps = fb
                return PredictOut(action_bps=fb, raw_action_bps=raw, clipped=clipped, smoothed=smoothed, fallback_used=True)

        inference_state.last_action_bps = float(a)
        return PredictOut(action_bps=float(a), raw_action_bps=float(raw), clipped=clipped, smoothed=smoothed, fallback_used=False)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True, "ts": int(time.time())}

    @app.post("/predict", response_model=PredictOut)
    def predict(req: PredictIn) -> PredictOut:
        return predict_action(req.state, req.prev_action_bps, req.fallback_action_bps, http_state)

    @app.websocket("/ws")
    async def ws_endpoint(ws):
        ws_state = InferenceState(history=deque(maxlen=norm.window_size))
        await ws.accept()
        while True:
            msg = await ws.receive_json()
            req = PredictIn(**msg)
            out = predict_action(req.state, req.prev_action_bps, req.fallback_action_bps, ws_state)
            await ws.send_json(out.model_dump())

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
