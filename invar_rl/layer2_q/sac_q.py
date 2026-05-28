"""Stable-Baselines3 SAC subclass with twin quantile critics + CVaR-blend actor.

The subclass keeps the SB3 SAC base for: ``learn`` loop, replay buffer,
callbacks, save/load shell, deterministic seeding, ``ActionNoise``,
``VecNormalize`` hooks, and the standard squashed-Gaussian SAC actor on
the full observation. The action is the canonical 1-D exposure scalar;
the wrapper is unchanged.

The two changes vs canonical SAC:

1. The twin scalar Q-critics (and their Polyak targets) are replaced with
   :class:`invar_rl.layer2_q.quantile_critic.QuantileCritic` twins. Each
   critic predicts ``Nq`` quantile values of Q(s, a). The Bellman target
   is computed at the quantile level using clipped double-Q (elementwise
   min across the two critics).

2. The actor loss replaces SB3's ``ent_coef * log_prob - min(Q1, Q2)``
   scalar target with ``ent_coef * log_prob - (eta * q_mean + (1 - eta)
   * q_cvar)``, where ``q_mean`` and ``q_cvar`` are reduced over the
   quantile axis of the per-critic mean ``0.5 * (Q1 + Q2)``.

Both changes preserve SB3's action-space contract (replay buffer stores
scaled actions in [-1, 1]; critics consume scaled actions).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.nn import functional as F

from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update

from invar_rl.layer2_q.config import QConfig
from invar_rl.layer2_q.cvar import cvar_from_quantiles
from invar_rl.layer2_q.quantile_critic import (
    QuantileCritic,
    huber_quantile_loss,
)


class _TrainStats:
    """Rolling mean over Q-specific scalars reported each ``train()`` call."""

    def __init__(self) -> None:
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def add(self, key: str, value: float) -> None:
        self._sums[key] = self._sums.get(key, 0.0) + float(value)
        self._counts[key] = self._counts.get(key, 0) + 1

    def mean(self, key: str) -> float:
        c = self._counts.get(key, 0)
        if c <= 0:
            return 0.0
        return self._sums[key] / c

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()


class SACQ(SAC):
    """SB3 SAC subclass with twin QuantileCritics + CVaR-blend actor loss.

    The SB3 actor (squashed-Gaussian on full obs) is unchanged. The SB3
    twin-Q critic (and its Polyak target) are replaced after parent
    construction with QuantileCritic twins; the critic optimiser is
    re-bound to the new parameters.

    Args:
        policy: SB3 policy spec, e.g. ``"MlpPolicy"``.
        env: A gymnasium env with a 1-D Box observation and a 1-D Box
            action space.
        q_config: :class:`QConfig` controlling Q-specific defaults.
        learning_starts: SB3 SAC warm-up steps before train() runs.
        train_freq: SB3 train_freq passed through.
        gradient_steps: SB3 gradient_steps passed through.
        verbose: SB3 verbosity passed through.
        seed: SB3 seed passed through.
        device: SB3 device string ("auto"/"cpu"/"cuda").
    """

    def __init__(
        self,
        policy,
        env,
        q_config: QConfig,
        learning_starts: int = 100,
        train_freq: int = 1,
        gradient_steps: int = 1,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=float(q_config.learning_rate),
            buffer_size=int(q_config.buffer_size),
            learning_starts=int(learning_starts),
            batch_size=int(q_config.batch_size),
            tau=float(q_config.polyak_tau),
            gamma=float(q_config.gamma),
            train_freq=int(train_freq),
            gradient_steps=int(gradient_steps),
            seed=seed,
            verbose=int(verbose),
            device=device,
            policy_kwargs={"net_arch": list(q_config.actor_hidden)},
            **kwargs,
        )
        self.q_config = q_config
        self._n_quantiles = int(q_config.n_quantiles)
        self._alpha_cvar = float(q_config.alpha_cvar)
        self._eta_blend = float(q_config.eta_blend)
        self._train_stats = _TrainStats()
        # Replace SB3's scalar twin-Q critics with QuantileCritic twins.
        # Done AFTER the parent _setup_model + _create_aliases pass, so
        # we overwrite both self.policy.critic and the alias self.critic.
        self._build_quantile_critics()

    # ------------------------------------------------------------------
    # Critic swap.
    # ------------------------------------------------------------------
    def _build_quantile_critics(self) -> None:
        """Replace SB3 twin scalar critics with QuantileCritic twins."""
        obs_space = self.observation_space
        act_space = self.action_space
        if obs_space.shape is None or len(obs_space.shape) != 1:
            raise ValueError(
                f"SACQ expects a 1-D observation space; "
                f"got shape {obs_space.shape}"
            )
        if act_space.shape is None or len(act_space.shape) != 1:
            raise ValueError(
                f"SACQ expects a 1-D action space; got shape {act_space.shape}"
            )
        obs_dim = int(obs_space.shape[0])
        action_dim = int(act_space.shape[0])
        if action_dim != 1:
            raise ValueError(
                f"SACQ assumes a scalar exposure action; got "
                f"action_dim={action_dim}"
            )
        critic = QuantileCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_quantiles=self._n_quantiles,
            hidden=tuple(self.q_config.critic_hidden),
            n_critics=2,
        ).to(self.device)
        critic_target = QuantileCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_quantiles=self._n_quantiles,
            hidden=tuple(self.q_config.critic_hidden),
            n_critics=2,
        ).to(self.device)
        critic_target.load_state_dict(critic.state_dict())
        for p in critic_target.parameters():
            p.requires_grad_(False)
        # SB3 SAC._create_aliases binds self.critic / self.critic_target
        # as direct attributes on the SAC agent that mirror policy
        # attributes. We must overwrite BOTH the policy attribute and the
        # agent alias, otherwise downstream SB3 code paths (and our own
        # train()) silently pick up the original scalar critic.
        self.policy.critic = critic
        self.policy.critic_target = critic_target
        self.critic = critic
        self.critic_target = critic_target
        # Optimiser plumbing: rebuild Adam on the new critic params.
        critic_parameters = list(critic.parameters())
        self.policy.critic.optimizer = self.policy.optimizer_class(
            critic_parameters,
            lr=float(self.q_config.learning_rate),
            **self.policy.optimizer_kwargs,
        )
        # batch_norm_stats are computed in parent _setup_model from the
        # scalar critic; QuantileCritic has no BatchNorm so the stats
        # lists are empty under the new module. Rebuild them as empty.
        from stable_baselines3.common.utils import get_parameters_by_name
        self.batch_norm_stats = get_parameters_by_name(
            self.critic, ["running_"]
        )
        self.batch_norm_stats_target = get_parameters_by_name(
            self.critic_target, ["running_"]
        )

    def _create_aliases(self) -> None:
        """Parent calls this during _setup_model; keep the default behaviour.

        At parent _setup_model time the policy still owns the scalar
        ContinuousCritic; we let SB3 alias them as usual so the parent's
        downstream setup (batch_norm_stats, target_entropy) runs against
        a fully-shaped critic. Our :meth:`_build_quantile_critics`
        overrides the aliases AFTER parent construction is complete.
        """
        super()._create_aliases()

    # ------------------------------------------------------------------
    # Train: quantile Bellman + CVaR-blend actor loss.
    # ------------------------------------------------------------------
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """Run ``gradient_steps`` SAC updates with the quantile critics.

        Critic update:
            - Sample (obs, action, reward, next_obs, done) from the
              replay buffer; actions are scaled to [-1, 1] per SB3.
            - Sample next-action a' from the actor on next_obs; clipped
              double-Q at the QUANTILE level: take the elementwise min
              of Q1_target(next_obs, a') and Q2_target(next_obs, a')
              along the quantile axis, then subtract ent_coef *
              log_prob(a' | next_obs) per the SAC entropy term.
            - Target distribution y = reward + gamma * (1 - done) *
              (min_quantiles - ent_coef * log_prob); shape [B, Nq].
            - Per-critic huber-quantile loss against the [B, Nq] target;
              sum the two losses and step the critic optimiser.

        Actor update:
            - Sample a fresh action a_pi from the actor on obs; compute
              both critics' quantile predictions at (obs, a_pi); take
              the mean across the two critics; reduce to q_mean (over
              quantiles) and q_cvar (over the lower-alpha-fraction).
            - Actor loss = ent_coef * log_prob - (eta * q_mean +
              (1 - eta) * q_cvar), averaged over the batch.
            - Step the actor optimiser.

        Polyak update:
            - polyak_update on the live -> target critics with self.tau.
        """
        self.policy.set_training_mode(True)
        self._train_stats.reset()

        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        for gradient_step in range(int(gradient_steps)):
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else float(self.gamma)
            )
            obs = replay_data.observations.float()
            action = replay_data.actions.float()
            next_obs = replay_data.next_observations.float()
            reward = replay_data.rewards.float().reshape(-1, 1)
            done = replay_data.dones.float().reshape(-1, 1)
            if obs.dim() != 2:
                obs = obs.reshape(obs.shape[0], -1)
                next_obs = next_obs.reshape(next_obs.shape[0], -1)
            if action.dim() != 2:
                action = action.reshape(action.shape[0], -1)

            # ---- pi(s) for entropy coefficient + actor update ---------
            actions_pi, log_prob_pi = self.actor.action_log_prob(obs)
            log_prob_pi = log_prob_pi.reshape(-1, 1)

            # ---- alpha (entropy coefficient) --------------------------
            ent_coef_loss = None
            if (
                self.ent_coef_optimizer is not None
                and self.log_ent_coef is not None
            ):
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(
                    self.log_ent_coef
                    * (log_prob_pi + self.target_entropy).detach()
                ).mean()
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                ent_coef = torch.exp(self.log_ent_coef.detach())
            else:
                ent_coef = self.ent_coef_tensor

            # ---- Critic update (quantile Bellman) ---------------------
            taus = self.critic.taus
            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(
                    next_obs
                )
                # Twin target quantiles, each [B, Nq].
                next_q_tuple = self.critic_target(next_obs, next_actions)
                next_q_stack = torch.stack(next_q_tuple, dim=0)
                # Clipped double-Q at the quantile level: elementwise min
                # across the two critics, per sample, per quantile.
                next_q_min, _ = torch.min(next_q_stack, dim=0)
                # Subtract entropy term per the SAC target.
                next_q_min = next_q_min - ent_coef * next_log_prob.reshape(
                    -1, 1
                )
                target_q = (
                    reward
                    + (1.0 - done) * discounts * next_q_min
                )  # [B, Nq]
            current_q_tuple = self.critic(obs, action)
            critic_loss = sum(
                huber_quantile_loss(
                    pred=q_pred,
                    target=target_q,
                    taus=taus,
                    kappa=1.0,
                )
                for q_pred in current_q_tuple
            )
            self.policy.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.policy.critic.optimizer.step()

            # ---- Actor update (CVaR-blend) ----------------------------
            for p in self.critic.parameters():
                p.requires_grad_(False)
            q_pi_tuple = self.critic(obs, actions_pi)
            q_pi_avg = 0.5 * (q_pi_tuple[0] + q_pi_tuple[1])
            q_mean = q_pi_avg.mean(dim=-1)
            q_cvar = cvar_from_quantiles(q_pi_avg, alpha=self._alpha_cvar)
            actor_target = (
                self._eta_blend * q_mean
                + (1.0 - self._eta_blend) * q_cvar
            ).reshape(-1, 1)
            actor_loss = (ent_coef * log_prob_pi - actor_target).mean()
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            for p in self.critic.parameters():
                p.requires_grad_(True)

            # ---- Polyak update of target critics ---------------------
            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(),
                    self.critic_target.parameters(),
                    float(self.tau),
                )

            with torch.no_grad():
                self._train_stats.add(
                    "q/critic_loss", float(critic_loss.item())
                )
                self._train_stats.add(
                    "q/actor_loss", float(actor_loss.item())
                )
                self._train_stats.add(
                    "q/q_mean", float(q_mean.mean().item())
                )
                self._train_stats.add(
                    "q/q_cvar", float(q_cvar.mean().item())
                )
                self._train_stats.add(
                    "q/target_q_mean",
                    float(target_q.mean().item()),
                )
                self._train_stats.add(
                    "q/log_prob", float(log_prob_pi.mean().item())
                )
                self._train_stats.add(
                    "q/ent_coef", float(ent_coef.item())
                )
                if ent_coef_loss is not None:
                    self._train_stats.add(
                        "q/ent_coef_loss", float(ent_coef_loss.item())
                    )

        self._n_updates += int(gradient_steps)
        for key in (
            "q/critic_loss",
            "q/actor_loss",
            "q/q_mean",
            "q/q_cvar",
            "q/target_q_mean",
            "q/log_prob",
            "q/ent_coef",
            "q/ent_coef_loss",
        ):
            self.logger.record(key, self._train_stats.mean(key))
        self.logger.record(
            "train/n_updates", self._n_updates, exclude="tensorboard"
        )

    # ------------------------------------------------------------------
    # Convenience read-outs (used by drivers).
    # ------------------------------------------------------------------
    def q_train_stats(self) -> Dict[str, float]:
        return {
            "critic_loss": self._train_stats.mean("q/critic_loss"),
            "actor_loss": self._train_stats.mean("q/actor_loss"),
            "q_mean": self._train_stats.mean("q/q_mean"),
            "q_cvar": self._train_stats.mean("q/q_cvar"),
            "target_q_mean": self._train_stats.mean("q/target_q_mean"),
            "log_prob": self._train_stats.mean("q/log_prob"),
            "ent_coef": self._train_stats.mean("q/ent_coef"),
            "ent_coef_loss": self._train_stats.mean("q/ent_coef_loss"),
        }
