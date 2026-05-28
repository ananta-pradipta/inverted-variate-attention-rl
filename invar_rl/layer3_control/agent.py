"""Reinforcement-learning controllers for Layer 3.

The primary agent is recurrent PPO (an LSTM policy and value network via
sb3-contrib RecurrentPPO). The recurrence is required because the
environment is a partially observed Markov decision process: the latent
regime must be inferred from the observation history.

Reinforcement-learning baselines:

- Feedforward, non-recurrent PPO. This is the ablation that tests whether
  the recurrence, and therefore the partially observed framing, matters.
- Soft Actor-Critic (off-policy). See the note below.

Note on recurrent SAC: the spec asks for a recurrent Soft Actor-Critic
baseline, but the maintained stable-baselines3 ecosystem has no recurrent
SAC (sb3-contrib provides RecurrentPPO only; SAC in stable-baselines3 is
feedforward). A clean custom recurrent SAC is a substantial component and
is deliberately not implemented here under scope discipline; a feedforward
SAC is provided as the off-policy comparator and the recurrence question is
isolated by the recurrent-PPO versus feedforward-PPO contrast. Whether to
build a custom recurrent SAC is raised in the Phase 5 summary.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from invar_rl.common.config import Stage3Config

RL_METHODS = ("recurrent_ppo", "feedforward_ppo", "sac")


def build_agent(
    method: str, env: gym.Env, cfg: Stage3Config, seed: int
) -> Any:
    """Construct an untrained RL agent for ``method``.

    Args:
        method: One of ``RL_METHODS``.
        env: The exposure-control environment instance.
        cfg: Stage 3 configuration.
        seed: Integer seed for reproducibility.

    Returns:
        A stable-baselines3 / sb3-contrib model exposing ``learn`` and
        ``predict``.

    Raises:
        ValueError: If ``method`` is not a recognised RL method.
    """
    if method == "recurrent_ppo":
        from sb3_contrib import RecurrentPPO

        return RecurrentPPO(
            "MlpLstmPolicy",
            env,
            n_steps=cfg.n_steps,
            learning_rate=cfg.learning_rate,
            policy_kwargs={"lstm_hidden_size": cfg.recurrent_hidden},
            seed=seed,
            verbose=0,
        )
    if method == "feedforward_ppo":
        from stable_baselines3 import PPO

        return PPO(
            "MlpPolicy",
            env,
            n_steps=cfg.n_steps,
            learning_rate=cfg.learning_rate,
            seed=seed,
            verbose=0,
        )
    if method == "sac":
        from stable_baselines3 import SAC

        return SAC(
            "MlpPolicy",
            env,
            learning_rate=cfg.learning_rate,
            seed=seed,
            verbose=0,
        )
    raise ValueError(
        f"unknown RL method {method!r}, expected one of {RL_METHODS}"
    )
