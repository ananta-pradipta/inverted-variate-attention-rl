"""Stable-Baselines3 SAC subclass with the SIA Sparse Invariant Actor.

The subclass keeps the SB3 SAC base for: ``learn`` loop, replay buffer,
callbacks, save/load shell, deterministic seeding, ``ActionNoise`` and
``VecNormalize`` hooks. The SAC twin-Q critics (and their Polyak targets)
are the SB3 defaults; they keep the full observation as input, with no
bottleneck, which is the central asymmetric-AC design choice. The SB3
actor (Gaussian on a hidden vector) is constructed for the action-space
utilities (``policy.scale_action`` / ``policy.unscale_action``) but its
parameters are inert: both rollout prediction and the per-step actor
optimisation are routed through
:class:`invar_rl.layer2_sia.sparse_actor.SparseInvariantActor`.

Action-space contract (matches SB3's native squashed-Gaussian SAC):

- The SIA actor's forward returns a tanh-squashed 1-D action in [-1, 1]
  together with its log-prob and an aux dict carrying the latent (mu,
  logvar, gates).
- In :meth:`predict` we unscale that to [exposure_low, exposure_high]
  (e.g. [0, 1.5]) via :meth:`policy.unscale_action` BEFORE returning,
  so the env step sees the un-normalised exposure.
- SB3's :meth:`_sample_action` then runs ``buffer_action =
  policy.scale_action(action)``, putting the action back into [-1, 1]
  for replay-buffer storage. Critic Q forwards on
  ``replay_buffer.actions`` therefore use [-1, 1], and so do all
  ``self.critic(obs, action_squashed)`` calls inside :meth:`train`.
  There is a SINGLE, coherent action axis for the critic.

Auxiliary loss (KL bottleneck + gate L1 + regime invariance) is computed
on the LATENT z, not on the action. The action's log-prob is the standard
SB3 squashed-Gaussian log-prob of a 1-D action, so SAC's entropy auto-
tuning behaves identically to canonical SAC.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.utils import polyak_update

from invar_rl.layer2_sia.aux_loss import actor_aux_loss
from invar_rl.layer2_sia.config import SIAConfig
from invar_rl.layer2_sia.sparse_actor import (
    SparseInvariantActor,
    resolve_dims,
)


class _TrainStats:
    """Rolling mean over SIA-specific scalars reported each ``train()`` call."""

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


class SACSIA(SAC):
    """SB3 SAC subclass with a SparseInvariantActor + full-info SB3 critics.

    The actor (SB3 default Gaussian) is constructed by the parent class but
    its parameters are inert; the SIA actor drives both predict() and the
    actor-update half of train(). The critics + their targets are SB3
    defaults (net_arch=[256, 256]) and take gradients normally during the
    critic update.

    Args:
        policy: Standard SB3 policy spec, e.g. ``"MlpPolicy"``.
        env: A gymnasium env with a 1-D Box observation and a 1-D Box
            action space of bounds [exposure_low, exposure_high].
        sia_config: :class:`SIAConfig` controlling SIA defaults.
        macro_dim: Dimension of the raw macro_encoding block inside the
            observation.
        l1_uncertainty: 1 if the obs carries an extra L1-uncertainty
            scalar in its tail, else 0. SP500 / NDX / NBI default to 0.
        regime_lookup: Deprecated; kept for backwards compatibility. Use
            ``regime_label=True`` plus
            :class:`invar_rl.layer2_sia.env_wrapper.RegimeLabelEnv`
            instead. If both are provided, ``regime_label=True`` takes
            precedence and the lookup is ignored. When neither is
            provided the invariance term in the auxiliary loss is zero
            on every minibatch ("I" no-op).
        regime_label: If True, the obs tail carries a trailing
            k-means-8 cluster id (float 0..7) appended by
            :class:`~invar_rl.layer2_sia.env_wrapper.RegimeLabelEnv`. The
            actor strips this column from its gate and encoder inputs;
            the SACSIA train step reads it as the per-row group id for
            the regime-invariance penalty in the actor aux loss.
        learning_starts: SB3 SAC warm-up steps before train() runs.
        verbose: SB3 verbosity passed through.
        seed: SB3 seed passed through.
        device: SB3 device string ("auto"/"cpu"/"cuda").
    """

    def __init__(
        self,
        policy,
        env,
        sia_config: SIAConfig,
        macro_dim: int,
        l1_uncertainty: int = 0,
        regime_lookup: Optional[Dict[int, int]] = None,
        regime_label: bool = False,
        learning_starts: int = 100,
        train_freq: int = 1,
        gradient_steps: int = 1,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        # Critic net_arch fixed at SB3 default [256, 256] for parity with
        # canonical SAC (see audit B3). sia_config.critic_hidden is kept
        # in the config for back-compat but is intentionally NOT consumed
        # here.
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=float(sia_config.learning_rate),
            buffer_size=int(sia_config.buffer_size),
            learning_starts=int(learning_starts),
            batch_size=int(sia_config.batch_size),
            tau=float(sia_config.polyak_tau),
            gamma=float(sia_config.gamma),
            train_freq=int(train_freq),
            gradient_steps=int(gradient_steps),
            seed=seed,
            verbose=int(verbose),
            device=device,
            policy_kwargs={"net_arch": [256, 256]},
            **kwargs,
        )
        self.sia_config = sia_config
        self._macro_dim = int(macro_dim)
        self._l1_uncertainty = int(l1_uncertainty)
        self._regime_lookup = regime_lookup
        self._regime_label = bool(regime_label)
        self._build_sia_actor()
        self._train_stats = _TrainStats()

    # ------------------------------------------------------------------
    # Build + inspect helpers.
    # ------------------------------------------------------------------
    def _build_sia_actor(self) -> None:
        """Construct the SIA actor + optimiser on top of SB3's setup."""
        obs_space = self.observation_space
        act_space = self.action_space
        if obs_space.shape is None or len(obs_space.shape) != 1:
            raise ValueError(
                f"SACSIA expects a 1-D observation space; "
                f"got shape {obs_space.shape}"
            )
        if act_space.shape is None or len(act_space.shape) != 1:
            raise ValueError(
                f"SACSIA expects a 1-D action space; "
                f"got shape {act_space.shape}"
            )
        obs_dim = int(obs_space.shape[0])
        action_dim = int(act_space.shape[0])
        if action_dim != 1:
            raise ValueError(
                f"SACSIA assumes a scalar exposure action; got "
                f"action_dim={action_dim}"
            )
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._exposure_low = float(act_space.low[0])
        self._exposure_high = float(act_space.high[0])

        dims = resolve_dims(
            obs_dim=obs_dim,
            macro_dim=self._macro_dim,
            l1_uncertainty=self._l1_uncertainty,
            macro_small_dim=16,
            regime_label=int(self._regime_label),
        )
        self.sia_actor = SparseInvariantActor(
            dims=dims,
            latent_dim=int(self.sia_config.latent_dim),
            actor_hidden=tuple(self.sia_config.actor_hidden),
            exposure_high=float(self._exposure_high),
            sparse_gates=bool(self.sia_config.sparse_gates),
        ).to(self.device)
        self.sia_actor_optim = torch.optim.Adam(
            self.sia_actor.parameters(),
            lr=float(self.sia_config.learning_rate),
        )
        # Phase 4 no_a ablation: rebuild the SB3 twin-Q critic so it
        # consumes the actor's post-gate ``actor_in`` bottleneck instead
        # of the full observation. The critic is rebuilt AFTER SB3's
        # _setup_model has already constructed a full-obs critic; we
        # overwrite self.critic / self.critic_target and re-bind the
        # critic optimiser. The replay buffer still stores the raw obs;
        # the train() loop calls self.sia_actor.encode() on the sampled
        # obs to produce the bottleneck before each critic forward.
        if not bool(self.sia_config.asymmetric_critic):
            self._rebuild_critic_on_bottleneck()
        self._asymmetric_critic = bool(self.sia_config.asymmetric_critic)

    def _rebuild_critic_on_bottleneck(self) -> None:
        """Replace the SB3 twin-Q critic with one keyed on actor_in_dim.

        SB3's :class:`~stable_baselines3.common.policies.ContinuousCritic`
        takes ``(features_dim + action_dim) -> 1`` MLPs; we instantiate a
        fresh pair with a ``FlattenExtractor`` over a bottleneck Box of
        width ``actor_in_dim``. The Polyak target is built the same way
        and seeded from the new critic's state_dict so the target track
        starts equal to the live critic, matching SB3 convention. The
        critic optimiser is re-bound to the new parameters.
        """
        actor_in_dim = int(self.sia_actor.actor_in_dim)
        bottleneck_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(actor_in_dim,),
            dtype=np.float32,
        )
        critic_kwargs: Dict[str, Any] = {
            "observation_space": bottleneck_space,
            "action_space": self.action_space,
            "net_arch": [256, 256],
            "n_critics": 2,
            "share_features_extractor": False,
            "activation_fn": torch.nn.ReLU,
            "normalize_images": True,
        }
        critic = ContinuousCritic(
            features_extractor=FlattenExtractor(bottleneck_space),
            features_dim=actor_in_dim,
            **critic_kwargs,
        ).to(self.device)
        critic_target = ContinuousCritic(
            features_extractor=FlattenExtractor(bottleneck_space),
            features_dim=actor_in_dim,
            **critic_kwargs,
        ).to(self.device)
        critic_target.load_state_dict(critic.state_dict())
        critic_target.set_training_mode(False)
        # SB3 SAC._create_aliases binds self.critic / self.critic_target
        # as DIRECT attributes on the SAC agent (not properties), and they
        # are independent references to self.policy.critic /
        # self.policy.critic_target. We must overwrite BOTH the policy
        # attribute and the agent alias, otherwise train()'s
        # ``self.critic(obs, action)`` still resolves to the original
        # full-obs critic via the alias and silently shape-mismatches.
        self.policy.critic = critic
        self.policy.critic_target = critic_target
        self.critic = critic
        self.critic_target = critic_target
        # SB3 optimiser plumbing: rebuild Adam on the new critic params.
        critic_parameters = list(critic.parameters())
        self.policy.critic.optimizer = self.policy.optimizer_class(
            critic_parameters,
            lr=float(self.sia_config.learning_rate),
            **self.policy.optimizer_kwargs,
        )

    def _critic_obs(
        self, obs: torch.Tensor, detach_encoder: bool = False
    ) -> torch.Tensor:
        """Project an obs minibatch to the critic's input space.

        When ``asymmetric_critic`` is True (default) this is the identity.
        When False the critic was rebuilt on the bottleneck so we route
        the obs through the SIA actor's deterministic encode pipeline and
        return the ``actor_in`` tensor (post-gate, post-macro-projection,
        pre-z).

        Args:
            obs: float tensor of shape (B, obs_dim).
            detach_encoder: if True, the returned tensor is detached from
                the autograd graph, so the critic loss cannot push
                gradients back into the SIA actor's gate / macro_proj
                parameters. Used on the critic-update branch of train().
                The actor-update branch keeps detach=False so the actor
                loss can pressure the gates through the critic.
        """
        if self._asymmetric_critic:
            return obs
        _, _, _, actor_in = self.sia_actor.encode(obs)
        if detach_encoder:
            actor_in = actor_in.detach()
        return actor_in

    # ------------------------------------------------------------------
    # Inference: replace SB3's policy predict with the SIA actor.
    # ------------------------------------------------------------------
    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple]]:
        """Return the un-scaled exposure as the action.

        The SIA actor's raw output is a tanh-squashed scalar in [-1, 1];
        we then unscale to [exposure_low, exposure_high] via the SB3
        policy's ``unscale_action`` so the env step receives the same
        action range it would receive from canonical SAC.
        """
        obs_t = self._to_obs_tensor(observation)
        self.sia_actor.eval()
        with torch.no_grad():
            action_squashed, _, _ = self.sia_actor(
                obs_t, deterministic=bool(deterministic)
            )
        self.sia_actor.train()
        scaled_np = action_squashed.detach().cpu().numpy().astype(np.float32)
        # Numerical guard against tanh outputs sitting fractionally outside
        # [-1, 1] (mostly impossible, but safer for SB3's unscale assertion).
        scaled_np = np.clip(scaled_np, -1.0, 1.0)
        action_np = self.policy.unscale_action(scaled_np).astype(np.float32)
        if action_np.shape[0] == 1 and np.asarray(observation).ndim == 1:
            action_np = action_np.reshape(self._action_dim)
        action_np = np.clip(
            action_np, self._exposure_low, self._exposure_high
        )
        return action_np, state

    def _to_obs_tensor(self, observation) -> torch.Tensor:
        obs_np = np.asarray(observation, dtype=np.float32)
        if obs_np.ndim == 1:
            obs_np = obs_np.reshape(1, -1)
        return torch.from_numpy(obs_np).to(self.device)

    # ------------------------------------------------------------------
    # Train: SAC actor loss with SIA actor + standard SAC critic update.
    # ------------------------------------------------------------------
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """Run ``gradient_steps`` SAC updates with the SIA actor.

        All critic forwards use scaled actions in [-1, 1]; this is the
        SB3 replay-buffer convention and the SIA actor's native output
        range. The actor loss is the standard SAC reparameterised
        gradient ``alpha * log_prob - min(Q1, Q2)`` evaluated on a fresh
        sample from the SIA actor, plus the SIA auxiliary loss (KL on
        latent + gate L1 + regime invariance).
        """
        self.policy.set_training_mode(True)
        self.sia_actor.train()
        self._train_stats.reset()

        for _ in range(int(gradient_steps)):
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
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

            # ---- pi(s) for entropy coefficient + actor update --------
            # Compute once with grad; reuse for both ent_coef loss and
            # actor loss so the second SIA forward in the old code path
            # is gone (audit gradient-flow caveat).
            action_pi, log_prob_pi, aux = self.sia_actor(
                obs, deterministic=False
            )
            log_prob_pi = log_prob_pi.reshape(-1, 1)

            # ---- alpha (entropy coefficient) -------------------------
            if self.ent_coef_optimizer is not None:
                ent_coef_loss = -(
                    self.log_ent_coef
                    * (log_prob_pi + self.target_entropy).detach()
                ).mean()
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                ent_coef = self.log_ent_coef.exp().detach()
            else:
                ent_coef = torch.tensor(float(self.ent_coef), device=self.device)

            # ---- Critic update --------------------------------------
            with torch.no_grad():
                next_action, next_log_prob, _ = self.sia_actor(
                    next_obs, deterministic=False
                )
                next_critic_obs = self._critic_obs(
                    next_obs, detach_encoder=True
                )
                next_q_values = torch.cat(
                    self.critic_target(next_critic_obs, next_action), dim=1
                )
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                next_q_values = (
                    next_q_values
                    - ent_coef * next_log_prob.reshape(-1, 1)
                )
                target_q_values = (
                    reward + (1.0 - done) * float(self.gamma) * next_q_values
                )
            current_critic_obs = self._critic_obs(obs, detach_encoder=True)
            current_q_values = self.critic(current_critic_obs, action)
            critic_loss = 0.5 * sum(
                F.mse_loss(c_q, target_q_values) for c_q in current_q_values
            )
            self.policy.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.policy.critic.optimizer.step()

            # ---- Actor update --------------------------------------
            # Freeze critics during the actor step; standard SAC trick.
            for p in self.critic.parameters():
                p.requires_grad_(False)

            self.sia_actor_optim.zero_grad()
            actor_critic_obs = self._critic_obs(obs, detach_encoder=False)
            q_values_pi = torch.cat(
                self.critic(actor_critic_obs, action_pi), dim=1
            )
            min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
            sac_actor_loss = (
                ent_coef * log_prob_pi - min_qf_pi
            ).mean()

            group_ids = self._group_ids_for_batch(replay_data)
            terms, aux_total = actor_aux_loss(
                aux=aux,
                group_ids=group_ids,
                beta_kl=float(self.sia_config.beta_kl),
                lambda_gate=float(self.sia_config.lambda_gate),
                lambda_inv=float(self.sia_config.lambda_inv),
            )
            total_actor_loss = sac_actor_loss + aux_total
            total_actor_loss.backward()
            self.sia_actor_optim.step()

            for p in self.critic.parameters():
                p.requires_grad_(True)

            # ---- Target critic Polyak update -----------------------
            polyak_update(
                self.critic.parameters(),
                self.critic_target.parameters(),
                float(self.tau),
            )

            with torch.no_grad():
                self._train_stats.add(
                    "sia/critic_loss", float(critic_loss.item())
                )
                self._train_stats.add(
                    "sia/actor_loss", float(sac_actor_loss.item())
                )
                self._train_stats.add(
                    "sia/aux_total", float(aux_total.detach().item())
                )
                self._train_stats.add("sia/aux_kl", float(terms.kl))
                self._train_stats.add("sia/aux_gate_l1", float(terms.gate_l1))
                self._train_stats.add("sia/aux_inv", float(terms.inv))
                gates_mean = aux["gates"].mean(dim=0).detach().cpu().numpy()
                for k in range(gates_mean.shape[0]):
                    self._train_stats.add(
                        f"sia/gate_{k}", float(gates_mean[k])
                    )
                self._train_stats.add(
                    "sia/gate_open_fraction",
                    float((aux["gates"] > 0.5).float().mean().item()),
                )
                self._train_stats.add(
                    "sia/mu_std",
                    float(aux["mu"].std(dim=0).mean().item()),
                )
                # Report exposure in the unscaled env units so the smoke
                # log is comparable to canonical SAC's exposure traces.
                exposure_env = (
                    0.5
                    * (action_pi + 1.0)
                    * (self._exposure_high - self._exposure_low)
                    + self._exposure_low
                )
                self._train_stats.add(
                    "sia/exposure_mean", float(exposure_env.mean().item())
                )
                self._train_stats.add(
                    "sia/exposure_std", float(exposure_env.std().item())
                )
                self._train_stats.add(
                    "sia/log_prob", float(log_prob_pi.mean().item())
                )
                self._train_stats.add(
                    "sia/ent_coef", float(ent_coef.item())
                )

        self._n_updates += int(gradient_steps)
        for key in (
            "sia/critic_loss", "sia/actor_loss", "sia/aux_total",
            "sia/aux_kl", "sia/aux_gate_l1", "sia/aux_inv",
            "sia/gate_0", "sia/gate_1", "sia/gate_2", "sia/gate_3", "sia/gate_4",
            "sia/gate_open_fraction", "sia/mu_std",
            "sia/exposure_mean", "sia/exposure_std",
            "sia/log_prob", "sia/ent_coef",
        ):
            self.logger.record(key, self._train_stats.mean(key))
        self.logger.record(
            "train/n_updates", self._n_updates, exclude="tensorboard"
        )

    def _group_ids_for_batch(
        self, replay_data
    ) -> Optional[torch.Tensor]:
        """Resolve per-row group ids for the regime-invariance penalty.

        Two wiring paths:

        - ``regime_label=True`` (recommended, from Phase 1 post-fix +
          regime-wired): the env wrapper
          :class:`~invar_rl.layer2_sia.env_wrapper.RegimeLabelEnv`
          appends the k-means-8 cluster id as the LAST obs column. We
          read that column directly and cast to long; no Python lookup
          per row, and the id is consistent across the replay buffer.
        - ``regime_lookup`` (deprecated fallback): expects the obs tail
          to carry an integer day_idx in its LAST column AND a
          ``{day_idx -> group_id}`` lookup; mostly kept so existing
          callers do not break.

        When neither path is wired, returns ``None`` and the invariance
        term in the actor aux loss is zero on every minibatch.
        """
        if self._regime_label:
            obs = replay_data.observations.float()
            if obs.dim() != 2:
                obs = obs.reshape(obs.shape[0], -1)
            return obs[:, -1].detach().long()
        if self._regime_lookup is None:
            return None
        obs = replay_data.observations.float()
        if obs.dim() != 2:
            obs = obs.reshape(obs.shape[0], -1)
        day_col = obs[:, -1].detach().cpu().numpy().astype(np.int64)
        ids = np.zeros(day_col.shape[0], dtype=np.int64)
        for i, d in enumerate(day_col):
            ids[i] = int(self._regime_lookup.get(int(d), 0))
        return torch.from_numpy(ids).to(self.device).long()

    # ------------------------------------------------------------------
    # Convenience read-outs (used by drivers and tests).
    # ------------------------------------------------------------------
    def sia_train_stats(self) -> Dict[str, float]:
        return {
            "critic_loss": self._train_stats.mean("sia/critic_loss"),
            "actor_loss": self._train_stats.mean("sia/actor_loss"),
            "aux_total": self._train_stats.mean("sia/aux_total"),
            "aux_kl": self._train_stats.mean("sia/aux_kl"),
            "aux_gate_l1": self._train_stats.mean("sia/aux_gate_l1"),
            "aux_inv": self._train_stats.mean("sia/aux_inv"),
            "gate_0": self._train_stats.mean("sia/gate_0"),
            "gate_1": self._train_stats.mean("sia/gate_1"),
            "gate_2": self._train_stats.mean("sia/gate_2"),
            "gate_3": self._train_stats.mean("sia/gate_3"),
            "gate_4": self._train_stats.mean("sia/gate_4"),
            "gate_open_fraction": self._train_stats.mean(
                "sia/gate_open_fraction"
            ),
            "mu_std": self._train_stats.mean("sia/mu_std"),
            "exposure_mean": self._train_stats.mean("sia/exposure_mean"),
            "exposure_std": self._train_stats.mean("sia/exposure_std"),
            "log_prob": self._train_stats.mean("sia/log_prob"),
            "ent_coef": self._train_stats.mean("sia/ent_coef"),
        }
