"""DDPG / TD3 on LunarLanderContinuous: overestimation bias (Q vs MC return), with or without LayerNorm."""

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import rl_mind.envs  # noqa: F401
from rl_mind.collectors import TransitionCollector
from rl_mind.core import Action, Actor
from rl_mind.data import ReplayBuffer, Transitions
from rl_mind.env import VecEnv
from rl_mind.nn import soft_update


@dataclass(frozen=True)
class Config:
    algo: str = "td3"
    layer_norm: bool = False
    depth: int = 2
    width: int = 256
    seed: int = 0

    env_name: str = "LunarLanderContinuous-v3"
    max_steps: int = 200_000
    learning_starts: int = 10_000
    buffer_size: int = 1_000_000
    batch_size: int = 256
    utd: int = 1
    gamma: float = 0.99
    tau: float = 0.005
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    action_noise: float = 0.1
    policy_delay: int = 2
    target_noise: float = 0.2
    target_noise_clip: float = 0.5

    eval_interval: int = 10_000
    n_eval_episodes: int = 10
    # states kept before a truncation
    mc_horizon: int = 300


def mlp(sizes: list[int], layer_norm: bool, out_act: nn.Module | None = None):
    layers: list[nn.Module] = []
    for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(n_in, n_out))
        if i < len(sizes) - 2:
            if layer_norm:
                layers.append(nn.LayerNorm(n_out))
            layers.append(nn.ReLU())
        elif out_act is not None:
            layers.append(out_act)
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden, layer_norm):
        super().__init__()
        self.model = mlp([obs_dim + act_dim, *hidden, 1], layer_norm)

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        return self.model(torch.cat([obs, action], dim=1)).squeeze(-1)


class DeterministicActor(Actor[Action]):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.model = mlp([obs_dim, *hidden, act_dim], False, nn.Tanh())

    def forward(self, obs: Tensor) -> Action:
        return Action(value=self.model(obs))


class GaussianNoise(Actor[Action]):
    def __init__(self, actor: Actor[Action], sigma: float):
        super().__init__()
        self.actor, self.sigma = actor, sigma

    def forward(self, obs: Tensor) -> Action:
        a = self.actor(obs).value
        return Action(value=(a + self.sigma * torch.randn_like(a)).clamp(-1, 1))

    def act(self, obs: Tensor) -> Tensor:
        return self.actor.act(obs)


def evaluate(cfg: Config, env: VecEnv, actor, critics) -> dict:
    obs = env.reset()
    n = env.num_envs
    alive = torch.ones(n, dtype=torch.bool)
    rewards, qs = [[] for _ in range(n)], [[] for _ in range(n)]
    terminated = [False] * n
    with torch.no_grad():
        while alive.any():
            action = actor.act(obs)
            q = torch.stack([c(obs, action) for c in critics])
            step = env.step(action)
            for i in range(n):
                if alive[i]:
                    rewards[i].append(float(step.reward[i]))
                    qs[i].append(q[:, i].numpy())
                    if step.done[i]:
                        alive[i] = False
                        terminated[i] = bool(step.terminated[i])
            obs = step.obs

    q_all, g_all = [], []
    for i in range(n):
        r = np.array(rewards[i])
        g = np.zeros_like(r)
        acc = 0.0
        for t in reversed(range(len(r))):
            acc = r[t] + cfg.gamma * acc
            g[t] = acc
        keep = len(r) if terminated[i] else max(0, len(r) - cfg.mc_horizon)
        q_all.append(np.array(qs[i])[:keep])
        g_all.append(g[:keep])
    q_all, g_all = np.concatenate(q_all), np.concatenate(g_all)
    q_min = q_all.min(axis=1)
    return {
        "return_mean": float(np.mean([sum(r) for r in rewards])),
        "returns": [float(sum(r)) for r in rewards],
        "ep_len": float(np.mean([len(r) for r in rewards])),
        "n_states": int(len(g_all)),
        "mc_mean": float(g_all.mean()),
        "q1_mean": float(q_all[:, 0].mean()),
        "bias_q1": float((q_all[:, 0] - g_all).mean()),
        "bias_target": float((q_min - g_all).mean()),
        "nbias_q1": float(((q_all[:, 0] - g_all) / (np.abs(g_all).mean() + 1e-8)).mean()),
    }


def run(cfg: Config) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    env = VecEnv(cfg.env_name, 1, seed=cfg.seed)
    eval_env = VecEnv(cfg.env_name, cfg.n_eval_episodes, seed=cfg.seed + 1000)
    od, ad = env.observation_dim, env.action_dim
    hidden = (cfg.width,) * cfg.depth
    td3 = cfg.algo == "td3"

    actor = DeterministicActor(od, ad, hidden)
    target_actor = copy.deepcopy(actor).requires_grad_(False)
    critics = [QNetwork(od, ad, hidden, cfg.layer_norm) for _ in range(2 if td3 else 1)]
    target_critics = [copy.deepcopy(c).requires_grad_(False) for c in critics]
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    critic_opt = torch.optim.Adam(
        [p for c in critics for p in c.parameters()], lr=cfg.lr_critic
    )

    collector = TransitionCollector(env, GaussianNoise(actor, cfg.action_noise))
    buffer = ReplayBuffer(cfg.buffer_size)
    history, n_updates, next_eval = [], 0, 0
    t0 = time.time()

    while collector.steps < cfg.max_steps:
        if collector.steps >= next_eval:
            res = evaluate(cfg, eval_env, actor, critics)
            res["step"] = collector.steps
            history.append(res)
            next_eval += cfg.eval_interval
            print(
                f"[{cfg.algo} ln={int(cfg.layer_norm)} d={cfg.depth} s={cfg.seed}] "
                f"step={collector.steps} ret={res['return_mean']:.1f} "
                f"Q={res['q1_mean']:.1f} G={res['mc_mean']:.1f} "
                f"bias={res['bias_q1']:.1f} ({time.time() - t0:.0f}s)",
                flush=True,
            )

        if collector.steps < cfg.learning_starts:
            buffer.add(_random_transitions(collector, ad))
            continue
        buffer.add(collector.collect(1))

        for _ in range(cfg.utd):
            n_updates += 1
            batch: Transitions[Action] = buffer.sample(cfg.batch_size)
            with torch.no_grad():
                if td3:
                    a2 = target_actor(batch.next_obs).value
                    eps = (torch.randn_like(a2) * cfg.target_noise).clamp(
                        -cfg.target_noise_clip, cfg.target_noise_clip
                    )
                    a2 = (a2 + eps).clamp(-1, 1)
                    q2 = torch.min(*[tc(batch.next_obs, a2) for tc in target_critics])
                else:
                    a2 = actor(batch.next_obs).value
                    q2 = target_critics[0](batch.next_obs, a2)
                y = batch.reward + cfg.gamma * (~batch.terminated).float() * q2
            critic_loss = sum(
                F.mse_loss(c(batch.obs, batch.action.value), y) for c in critics
            )
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            if not td3 or n_updates % cfg.policy_delay == 0:
                actor_loss = -critics[0](batch.obs, actor(batch.obs).value).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()
                for c, tc in zip(critics, target_critics):
                    soft_update(c, tc, cfg.tau)
                if td3:
                    soft_update(actor, target_actor, cfg.tau)

    res = evaluate(cfg, eval_env, actor, critics)
    res["step"] = collector.steps
    history.append(res)
    return {"config": asdict(cfg), "history": history, "time": time.time() - t0}


class _RandomActor(Actor[Action]):
    def __init__(self, act_dim):
        super().__init__()
        self.act_dim = act_dim

    def forward(self, obs: Tensor) -> Action:
        return Action(value=torch.rand(obs.shape[0], self.act_dim) * 2 - 1)


def _random_transitions(collector: TransitionCollector, act_dim: int):
    actor = collector.actor
    collector.actor = _RandomActor(act_dim)
    transitions = collector.collect(1)
    collector.actor = actor
    return transitions


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="td3")
    p.add_argument("--ln", type=int, default=0)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--utd", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--out", default="results")
    a = p.parse_args()
    torch.set_num_threads(1)
    cfg = Config(
        algo=a.algo, layer_norm=bool(a.ln), depth=a.depth, utd=a.utd,
        seed=a.seed, max_steps=a.steps,
    )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{a.algo}_ln{a.ln}_d{a.depth}_utd{a.utd}_s{a.seed}.json"
    result = run(cfg)
    (out / name).write_text(json.dumps(result))
