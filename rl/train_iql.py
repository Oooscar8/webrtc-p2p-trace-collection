#!/usr/bin/env python3
"""Train IQL with a Schaferct-style training loop on RL transitions.

This version assumes `tools/build_rl_dataset.py` already exports normalized
10-step historical states flattened to shape `[N, 50]` in feature-major order.

Compared with the previous trainer, the optimization flow now follows
`Schaferct/code/v14_iql.py` much more closely:

- replay buffer over offline transitions
- twin Q + value function
- asymmetric value regression
- target Q network soft update
- advantage-weighted behavior cloning actor update
- Gaussian policy actor by default
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR


BPS_IN_MBPS = 1_000_000.0
EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


TensorBatch = list[torch.Tensor]


@dataclass
class DatasetInfo:
    state_dim: int
    window_size: int
    state_cols: list[str]
    state_layout: str
    feature_mean: np.ndarray
    feature_std: np.ndarray
    observations_normalized: bool


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def to_python_scalar(value: np.ndarray | Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        if value.size == 1:
            return value.reshape(-1)[0].item() if hasattr(value.reshape(-1)[0], "item") else value.reshape(-1)[0]
        return value.tolist()
    return value


def to_string_list(value: np.ndarray | None, default: list[str]) -> list[str]:
    if value is None:
        return default
    return [str(item) for item in value.tolist()]


def extract_dataset_info(data: dict[str, np.ndarray]) -> DatasetInfo:
    observations = data["observations"].astype(np.float32)
    state_dim = int(to_python_scalar(data.get("state_dim"), observations.shape[1]))
    window_size = int(to_python_scalar(data.get("window_size"), 10))
    state_cols = to_string_list(data.get("state_cols"), ["send_bps", "recv_bps", "rtt_ms", "loss_rate", "jitter_ms"])
    state_layout = str(to_python_scalar(data.get("state_layout"), "feature_major"))
    feature_mean = np.asarray(data.get("feature_mean", np.zeros(len(state_cols), dtype=np.float32)), dtype=np.float32)
    feature_std = np.asarray(data.get("feature_std", np.ones(len(state_cols), dtype=np.float32)), dtype=np.float32)
    observations_normalized = bool(to_python_scalar(data.get("observations_normalized"), False))
    return DatasetInfo(
        state_dim=state_dim,
        window_size=window_size,
        state_cols=state_cols,
        state_layout=state_layout,
        feature_mean=feature_mean,
        feature_std=feature_std,
        observations_normalized=observations_normalized,
    )


def subset_dataset(data: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    subset: dict[str, np.ndarray] = {}
    transition_keys = {"observations", "actions", "rewards", "next_observations", "terminals"}
    for key, value in data.items():
        subset[key] = value[indices] if key in transition_keys else value
    return subset


def set_seed(seed: int, deterministic_torch: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1.0 - tau) * target_param.data + tau * source_param.data)


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u.pow(2))


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, buffer_size: int, device: str = "cpu"):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_dataset(self, data: dict[str, np.ndarray]) -> None:
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = int(data["observations"].shape[0])
        if n_transitions > self._buffer_size:
            raise ValueError("Replay buffer is smaller than the dataset you are trying to load")
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor((data["actions"] / BPS_IN_MBPS).reshape(-1, 1))
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"].reshape(-1, 1))
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"].reshape(-1, 1))
        self._size = n_transitions
        self._pointer = n_transitions

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._pointer, size=batch_size)
        return [
            self._states[indices],
            self._actions[indices],
            self._rewards[indices],
            self._next_states[indices],
            self._dones[indices],
        ]


class Squeeze(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims: list[int],
        activation_fn: type[nn.Module] = nn.ReLU,
        output_activation_fn: type[nn.Module] | None = None,
        squeeze_output: bool = False,
        dropout: float | None = None,
    ):
        super().__init__()
        if len(dims) < 2:
            raise ValueError("MLP requires at least two dims")
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 2):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(activation_fn())
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeeze_output=True")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action_mbps: float,
        hidden_dim: int = 256,
        dropout: float | None = None,
    ):
        super().__init__()
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action_mbps = float(max_action_mbps)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, 2)
        self.fc_mid = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.rb1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout) if dropout is not None else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
        )
        self.rb2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout) if dropout is not None else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
        )
        self.final = nn.Sequential(
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        obs: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        cell_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
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
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)).view(1, -1).expand(mean.shape[0], -1)
        out = torch.cat((mean, std), dim=-1)
        if add_sequence_dim:
            out = out.unsqueeze(0)
        return out, hidden_states, cell_states


class DeterministicPolicy(nn.Module):
    def __init__(self, state_dim: int, act_dim: int, max_action_mbps: float, hidden_dim: int = 256):
        super().__init__()
        self.max_action_mbps = float(max_action_mbps)
        self.net = MLP(
            [state_dim, hidden_dim, hidden_dim, act_dim],
            output_activation_fn=nn.Tanh,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.net(obs) * self.max_action_mbps, min=1e-5)


class TwinQ(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        dims = [state_dim + action_dim, hidden_dim, hidden_dim, 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.v = MLP([state_dim, hidden_dim, hidden_dim, 1], squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class ImplicitQLearning:
    def __init__(
        self,
        *,
        max_action_mbps: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        iql_tau: float,
        beta: float,
        max_steps: int,
        discount: float,
        tau: float,
        deterministic_actor: bool,
        grad_clip: float,
        device: str,
    ):
        self.max_action_mbps = max_action_mbps
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau
        self.total_it = 0
        self.deterministic_actor = deterministic_actor
        self.grad_clip = grad_clip

    def _clip_gradients(self, module: nn.Module) -> None:
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(module.parameters(), self.grad_clip)

    def _update_v(self, observations: torch.Tensor, actions: torch.Tensor, log_dict: dict[str, float]) -> torch.Tensor:
        with torch.no_grad():
            target_q = self.q_target(observations, actions)
        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = float(v_loss.item())
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self._clip_gradients(self.vf)
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: dict[str, float],
    ) -> None:
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        q1, q2 = self.qf.both(observations, actions)
        q_loss = (F.mse_loss(q1, targets) + F.mse_loss(q2, targets)) / 2.0
        log_dict["q_score"] = float(((torch.mean(q1) + torch.mean(q2)) / 2.0).item())
        log_dict["q_loss"] = float(q_loss.item())
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self._clip_gradients(self.qf)
        self.q_optimizer.step()
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: dict[str, float],
    ) -> None:
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        if self.deterministic_actor:
            policy_out = self.actor(observations)
            if policy_out.shape != actions.shape:
                raise RuntimeError("Deterministic actor output shape mismatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            policy_out, _, _ = self.actor(observations, None, None)
            mean = policy_out[:, : actions.shape[1]]
            std = policy_out[:, actions.shape[1] :]
            dist = Normal(mean, std)
            bc_losses = -dist.log_prob(actions).sum(-1)
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = float(policy_loss.item())
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self._clip_gradients(self.actor)
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train(self, batch: TensorBatch) -> dict[str, float]:
        self.total_it += 1
        observations, actions, rewards, next_observations, dones = batch
        log_dict: dict[str, float] = {}
        with torch.no_grad():
            next_v = self.vf(next_observations)
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        self._update_policy(adv, observations, actions, log_dict)
        return log_dict

    def state_dict(self) -> dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_target": self.q_target.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.qf.load_state_dict(state_dict["qf"])
        self.q_target.load_state_dict(state_dict.get("q_target", state_dict["qf"]))
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])
        self.total_it = int(state_dict["total_it"])


@torch.no_grad()
def evaluate_actor(
    actor: nn.Module,
    observations: torch.Tensor,
    actions: torch.Tensor,
    *,
    deterministic_actor: bool,
    chunk_size: int = 8192,
) -> dict[str, float]:
    if observations.shape[0] == 0:
        return {"val_mse": float("nan"), "val_nll": float("nan")}
    mse_total = 0.0
    nll_total = 0.0
    count = 0
    for start in range(0, observations.shape[0], chunk_size):
        end = min(start + chunk_size, observations.shape[0])
        obs_chunk = observations[start:end]
        act_chunk = actions[start:end]
        if deterministic_actor:
            mean = actor(obs_chunk)
            nll = torch.sum((mean - act_chunk) ** 2, dim=1)
        else:
            out, _, _ = actor(obs_chunk, None, None)
            mean = out[:, : act_chunk.shape[1]]
            std = out[:, act_chunk.shape[1] :]
            nll = -Normal(mean, std).log_prob(act_chunk).sum(-1)
        mse = torch.sum((mean - act_chunk) ** 2, dim=1)
        mse_total += float(mse.sum().item())
        nll_total += float(nll.sum().item())
        count += int(obs_chunk.shape[0])
    return {
        "val_mse": mse_total / max(count, 1),
        "val_nll": nll_total / max(count, 1),
    }


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def configure_torch_runtime(num_threads: int, num_interop_threads: int | None) -> dict[str, int]:
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    if num_interop_threads is not None and num_interop_threads > 0:
        torch.set_num_interop_threads(num_interop_threads)
    return {
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
    }


def build_artifact_metadata(
    *,
    dataset_info: DatasetInfo,
    max_action_mbps: float,
    deterministic_actor: bool,
    step: int,
    val_metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "state_cols": dataset_info.state_cols,
        "window_size": dataset_info.window_size,
        "state_dim": dataset_info.state_dim,
        "state_layout": dataset_info.state_layout,
        "feature_mean": dataset_info.feature_mean.tolist(),
        "feature_std": dataset_info.feature_std.tolist(),
        "observations_normalized": dataset_info.observations_normalized,
        "action_unit": "mbps",
        "action_scale_bps": BPS_IN_MBPS,
        "max_action_mbps": max_action_mbps,
        "max_action_bps": max_action_mbps * BPS_IN_MBPS,
        "actor_type": "deterministic" if deterministic_actor else "gaussian",
        "checkpoint_step": step,
        "val_mse": float(val_metrics["val_mse"]),
        "val_nll": float(val_metrics["val_nll"]),
    }


def save_checkpoint(
    *,
    checkpoint_dir: Path,
    trainer: ImplicitQLearning,
    dataset_info: DatasetInfo,
    max_action_mbps: float,
    deterministic_actor: bool,
    step: int,
    val_metrics: dict[str, float],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "qf": trainer.qf.state_dict(),
            "q_target": trainer.q_target.state_dict(),
            "vf": trainer.vf.state_dict(),
        },
        checkpoint_dir / "critics.pt",
    )
    torch.save(trainer.actor.state_dict(), checkpoint_dir / "actor.pt")
    torch.save(trainer.state_dict(), checkpoint_dir / "trainer.pt")
    with (checkpoint_dir / "norm.json").open("w", encoding="utf-8") as f:
        json.dump(
            build_artifact_metadata(
                dataset_info=dataset_info,
                max_action_mbps=max_action_mbps,
                deterministic_actor=deterministic_actor,
                step=step,
                val_metrics=val_metrics,
            ),
            f,
            ensure_ascii=False,
            indent=2,
        )


def append_checkpoint_record(
    *,
    metrics_path: Path,
    step: int,
    val_metrics: dict[str, float],
    checkpoint_dir: Path,
) -> None:
    record = {
        "step": step,
        "val_mse": float(val_metrics["val_mse"]),
        "val_nll": float(val_metrics["val_nll"]),
        "checkpoint_dir": str(checkpoint_dir),
        "actor_path": str(checkpoint_dir / "actor.pt"),
        "trainer_path": str(checkpoint_dir / "trainer.pt"),
    }
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="rl_dataset/transitions.npz")
    parser.add_argument("--outdir", type=str, default="models/iql")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--num-interop-threads", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--log-freq", type=int, default=1_000)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--iql-tau", type=float, default=0.7)
    parser.add_argument("--vf-lr", type=float, default=3e-4)
    parser.add_argument("--qf-lr", type=float, default=3e-4)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--actor-dropout", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--buffer-size", type=int, default=0)
    parser.add_argument("--max-action-mbps", type=float, default=0.0)
    parser.add_argument("--load-model", type=str, default="")
    parser.add_argument("--deterministic-actor", action="store_true")
    args = parser.parse_args()

    runtime_info = configure_torch_runtime(args.num_threads, args.num_interop_threads)
    set_seed(args.seed)
    device = resolve_device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = outdir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_metrics_path = outdir / "checkpoint_metrics.jsonl"

    dataset_path = Path(args.dataset)
    data = load_npz(dataset_path)
    dataset_info = extract_dataset_info(data)

    observations = data["observations"].astype(np.float32)
    actions_bps = data["actions"].astype(np.float32)

    num_samples = int(observations.shape[0])
    if num_samples < 1024:
        raise SystemExit(f"dataset too small: {num_samples}")
    if observations.shape[1] != dataset_info.state_dim:
        raise SystemExit(
            f"state_dim mismatch: observations={observations.shape[1]} metadata={dataset_info.state_dim}"
        )

    permutation = np.random.permutation(num_samples)
    num_train = max(1, int(num_samples * float(args.train_ratio)))
    num_train = min(num_train, num_samples - 1)
    train_idx = permutation[:num_train]
    val_idx = permutation[num_train:]
    if val_idx.size == 0:
        raise SystemExit("validation split is empty; reduce --train-ratio")

    train_data = subset_dataset(data, train_idx)
    val_actions_mbps = torch.tensor((actions_bps[val_idx] / BPS_IN_MBPS).reshape(-1, 1), dtype=torch.float32, device=device)
    val_observations = torch.tensor(observations[val_idx], dtype=torch.float32, device=device)

    buffer_size = int(args.buffer_size) if int(args.buffer_size) > 0 else int(train_idx.shape[0])
    replay_buffer = ReplayBuffer(dataset_info.state_dim, 1, buffer_size, str(device))
    replay_buffer.load_dataset(train_data)

    if args.max_action_mbps > 0:
        max_action_mbps = float(args.max_action_mbps)
    else:
        max_action_mbps = float(np.percentile(actions_bps[train_idx], 99.5) / BPS_IN_MBPS * 1.05)
        max_action_mbps = max(max_action_mbps, 1.0)

    q_network = TwinQ(dataset_info.state_dim, 1).to(device)
    v_network = ValueFunction(dataset_info.state_dim).to(device)
    actor: nn.Module
    if args.deterministic_actor:
        actor = DeterministicPolicy(dataset_info.state_dim, 1, max_action_mbps).to(device)
    else:
        actor = GaussianPolicy(
            dataset_info.state_dim,
            1,
            max_action_mbps,
            dropout=args.actor_dropout,
        ).to(device)

    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=args.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=args.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    trainer = ImplicitQLearning(
        max_action_mbps=max_action_mbps,
        actor=actor,
        actor_optimizer=actor_optimizer,
        q_network=q_network,
        q_optimizer=q_optimizer,
        v_network=v_network,
        v_optimizer=v_optimizer,
        iql_tau=args.iql_tau,
        beta=args.beta,
        max_steps=args.steps,
        discount=args.discount,
        tau=args.tau,
        deterministic_actor=args.deterministic_actor,
        grad_clip=args.grad_clip,
        device=str(device),
    )

    if args.load_model:
        trainer.load_state_dict(torch.load(args.load_model, map_location=device))

    with (outdir / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": str(dataset_path),
                "seed": args.seed,
                "device": str(device),
                "num_threads": runtime_info["torch_num_threads"],
                "num_interop_threads": runtime_info["torch_num_interop_threads"],
                "train_ratio": args.train_ratio,
                "steps": args.steps,
                "batch": args.batch,
                "eval_freq": args.eval_freq,
                "log_freq": args.log_freq,
                "discount": args.discount,
                "tau": args.tau,
                "beta": args.beta,
                "iql_tau": args.iql_tau,
                "vf_lr": args.vf_lr,
                "qf_lr": args.qf_lr,
                "actor_lr": args.actor_lr,
                "actor_dropout": args.actor_dropout,
                "grad_clip": args.grad_clip,
                "buffer_size": buffer_size,
                "max_action_mbps": max_action_mbps,
                "deterministic_actor": args.deterministic_actor,
                "state_dim": dataset_info.state_dim,
                "window_size": dataset_info.window_size,
                "state_layout": dataset_info.state_layout,
                "observations_normalized": dataset_info.observations_normalized,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("---------------------------------------")
    print("Training IQL with Schaferct-style updates")
    print(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "samples": num_samples,
                "train_samples": int(train_idx.shape[0]),
                "val_samples": int(val_idx.shape[0]),
                "state_dim": dataset_info.state_dim,
                "window_size": dataset_info.window_size,
                "max_action_mbps": max_action_mbps,
                "device": str(device),
                "num_threads": runtime_info["torch_num_threads"],
                "num_interop_threads": runtime_info["torch_num_interop_threads"],
                "actor_type": "deterministic" if args.deterministic_actor else "gaussian",
            },
            ensure_ascii=False,
        )
    )
    print("---------------------------------------")

    for step in range(1, int(args.steps) + 1):
        batch = replay_buffer.sample(args.batch)
        log_dict = trainer.train(batch)

        if step == 1 or step % args.log_freq == 0:
            print(
                f"step={step} q_loss={log_dict['q_loss']:.6f} "
                f"v_loss={log_dict['value_loss']:.6f} actor_loss={log_dict['actor_loss']:.6f} "
                f"q_score={log_dict['q_score']:.6f}",
                flush=True,
            )

        if step % args.eval_freq == 0:
            val_metrics = evaluate_actor(
                trainer.actor,
                val_observations,
                val_actions_mbps,
                deterministic_actor=args.deterministic_actor,
            )
            print(
                f"eval step={step} val_mse={val_metrics['val_mse']:.6f} val_nll={val_metrics['val_nll']:.6f}",
                flush=True,
            )
            checkpoint_dir = checkpoints_dir / f"checkpoint_{step}"
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                trainer=trainer,
                dataset_info=dataset_info,
                max_action_mbps=max_action_mbps,
                deterministic_actor=args.deterministic_actor,
                step=step,
                val_metrics=val_metrics,
            )
            append_checkpoint_record(
                metrics_path=checkpoint_metrics_path,
                step=step,
                val_metrics=val_metrics,
                checkpoint_dir=checkpoint_dir,
            )

    print(f"saved checkpoints to {checkpoints_dir}")


if __name__ == "__main__":
    main()
