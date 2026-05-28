"""Sparse Invariant Actor (SIA) module for InVAR-RL-SIA.

The actor partitions the canonical Layer 3 observation into five semantic
blocks (dispersion, wrapper_stats, macro, risk_state, l1_uncertainty),
projects macro down to a small encoding, gates four of the five blocks
through learned sigmoid gates, samples a KL-regularised latent z, and then
maps z to a tanh-squashed 1-D action in ``[-1, 1]``.

The actor follows SB3's SAC contract: the action is the tanh of a 1-D
Gaussian whose mean and log-std are functions of the latent z. SB3's
``policy.unscale_action`` then maps ``[-1, 1] -> [exposure_low,
exposure_high]`` (e.g. [0, 1.5]) for the env step + replay buffer storage.
This is the same contract SB3's :class:`SquashedDiagGaussianDistribution`
uses, with one twist: the 1-D action mean and log-std are produced from
the SIA latent z (a sparse, gated, KL-regularised bottleneck) rather than
directly from the observation.

Block layout, matching :mod:`invar_rl.layer3_control.observation` exactly:

- index 0           : Layer 1 score dispersion          (1-d) -- gated [0]
- index 1, 2        : wrapper stats (pred_vol, eff_N)   (2-d) -- gated [1:3]
- index 3, 4, 5, 6  : risk_state                        (4-d) -- NOT gated
- index 7..7+M-1    : macro_encoding (M = macro_dim)    (M-d) -> proj to 16-d, gated [3]
- (optional tail)   : l1_uncertainty                    (1-d) -- gated [4]; if
                                                              absent, the actor
                                                              feeds a zero so the
                                                              concat dim stays fixed

The risk_state block is intentionally not gated: the agent must always
see its own bookkeeping (rolling vol, drawdown, current exposure, days
since regime change). The gate vector emitted by ``gate_net`` has 5 logits
(one per gateable block); the risk_state gate slot is simply not used.

The actor's :meth:`forward` returns ``(action_squashed, log_prob, aux)``:

- ``action_squashed``: (B, 1) tensor in ``[-1, 1]``. SB3 will unscale this
  to the env's action space before the env step + buffer storage.
- ``log_prob``: (B,) tensor; the standard SB3-style tanh-squashed-Gaussian
  log-density of the realised action.
- ``aux``: dict with ``mu``, ``logvar``, and ``gates`` of the LATENT z.
  These are consumed by :mod:`invar_rl.layer2_sia.aux_loss`. The KL
  information bottleneck stays on the latent z, not on the action; the
  action distribution is the standard SAC squashed Gaussian.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn


_LOG_STD_MIN: float = -5.0
_LOG_STD_MAX: float = 2.0
_LOG_PROB_EPS: float = 1e-6


class MLP(nn.Module):
    """Plain multi-layer perceptron with ReLU activations.

    Args:
        in_dim: Input feature dimension; must be positive.
        hidden: Sequence of hidden-layer sizes; an empty sequence collapses
            the MLP to a single linear layer.
        out_dim: Output dimension; must be positive.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: Sequence[int],
        out_dim: int,
    ) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError("in_dim must be positive")
        if out_dim <= 0:
            raise ValueError("out_dim must be positive")
        sizes: List[int] = [int(in_dim)] + [int(h) for h in hidden] + [int(out_dim)]
        layers: List[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class SIADims:
    """Resolved observation slicing for the SparseInvariantActor.

    Attributes:
        total: Total observation dim seen by the agent.
        macro: Raw macro_encoding dim (M); projected to ``macro_small_dim``.
        l1_uncertainty: 1 if the obs tail carries an L1-uncertainty scalar,
            else 0. When 0 the actor still creates a zero tensor of width 1
            so the concat width and gate vector size stay fixed.
        macro_small_dim: Width after the macro projection (default 16).
        regime_label: 1 if the obs tail carries a trailing regime cluster
            id (integer 0..7 cast to float) for the regime-invariance
            penalty; else 0. The cluster id is NOT routed into the actor
            input (it is a group label, not a feature) but is exposed via
            :func:`_split_obs` so callers can read it for the aux loss.
    """

    total: int
    macro: int
    l1_uncertainty: int = 0
    macro_small_dim: int = 16
    regime_label: int = 0


def resolve_dims(
    obs_dim: int,
    macro_dim: int,
    l1_uncertainty: int = 0,
    macro_small_dim: int = 16,
    regime_label: int = 0,
) -> SIADims:
    """Build a :class:`SIADims` after validating the slicing arithmetic.

    The canonical observation layout (see module docstring) requires::

        obs_dim == 7 + macro_dim + l1_uncertainty + regime_label

    where 7 = 1 (dispersion) + 2 (wrapper_stats) + 4 (risk_state) and
    ``regime_label`` is 1 if the env wrapper has appended a trailing
    k-means-8 cluster id (cast to float) to the observation tail, else 0.
    """
    expected = 7 + int(macro_dim) + int(l1_uncertainty) + int(regime_label)
    if int(obs_dim) != expected:
        raise ValueError(
            f"obs_dim={obs_dim} inconsistent with macro_dim={macro_dim}, "
            f"l1_uncertainty={l1_uncertainty}, regime_label={regime_label}; "
            f"expected {expected}"
        )
    if macro_small_dim <= 0:
        raise ValueError("macro_small_dim must be positive")
    if int(regime_label) not in (0, 1):
        raise ValueError(
            f"regime_label must be 0 or 1; got {regime_label}"
        )
    return SIADims(
        total=int(obs_dim),
        macro=int(macro_dim),
        l1_uncertainty=int(l1_uncertainty),
        macro_small_dim=int(macro_small_dim),
        regime_label=int(regime_label),
    )


def _split_obs(
    obs: torch.Tensor, dims: SIADims
) -> Dict[str, torch.Tensor]:
    """Slice the flat observation into its semantic blocks.

    Returns a dict with keys: ``dispersion``, ``wrapper_stats``, ``macro``,
    ``risk_state``, ``l1_uncertainty``, and ``regime_label`` (always
    present). When ``dims.l1_uncertainty == 0``, the ``l1_uncertainty``
    value is a zero tensor of shape ``(B, 1)`` so the downstream concat
    width stays constant across universes. When ``dims.regime_label == 0``,
    the ``regime_label`` value is a zero (B, 1) placeholder; callers that
    need a real cluster id must construct dims with ``regime_label=1``.
    The regime_label is NOT routed into the actor input by the forward
    pass; it is consumed by the aux-loss group-id extractor only.
    """
    if obs.dim() != 2:
        raise ValueError(
            f"obs must be (B, D); got shape {tuple(obs.shape)}"
        )
    if obs.shape[1] != dims.total:
        raise ValueError(
            f"obs dim {obs.shape[1]} != configured total {dims.total}"
        )
    dispersion = obs[:, 0:1]
    wrapper_stats = obs[:, 1:3]
    risk_state = obs[:, 3:7]
    macro_start = 7
    macro_end = macro_start + dims.macro
    macro = obs[:, macro_start:macro_end]
    cursor = macro_end
    if dims.l1_uncertainty > 0:
        l1u = obs[:, cursor:cursor + dims.l1_uncertainty]
        cursor += dims.l1_uncertainty
    else:
        l1u = torch.zeros(
            (obs.shape[0], 1), dtype=obs.dtype, device=obs.device
        )
    if dims.regime_label > 0:
        regime_label = obs[:, cursor:cursor + 1]
    else:
        regime_label = torch.zeros(
            (obs.shape[0], 1), dtype=obs.dtype, device=obs.device
        )
    return {
        "dispersion": dispersion,
        "wrapper_stats": wrapper_stats,
        "macro": macro,
        "risk_state": risk_state,
        "l1_uncertainty": l1u,
        "regime_label": regime_label,
    }


class SparseInvariantActor(nn.Module):
    """Sparse, invariant, bottlenecked SAC actor for InVAR-RL-SIA.

    Forward pipeline (B is batch):
      1. Slice obs into 5 semantic blocks.
      2. ``gates = sigmoid(gate_net(obs))`` of shape (B, 5).
      3. Project macro to ``macro_small`` of width ``macro_small_dim``.
      4. Concatenate gated blocks + ungated risk_state.
      5. Latent: ``mu = mu_net(cat)``, ``logvar = logvar_net(cat).clamp``.
         ``z = mu + exp(0.5*logvar) * eps``, ``eps ~ N(0, I)``.
      6. Action: ``action_mu = action_mu_net(z)``,
         ``action_log_std = action_log_std_net(z).clamp(-5, 2)``.
         ``u = action_mu + exp(action_log_std) * eps_a``.
         ``action_squashed = tanh(u)`` of shape (B, 1).
      7. ``log_prob`` is the standard SB3 squashed-Gaussian log_prob.

    The latent z provides the SIA "sparse + invariant" pressure: gates +
    KL bottleneck are applied to z. The action distribution sitting on top
    of z is the canonical SAC tanh-squashed Gaussian, so SB3's existing
    plumbing (``scale_action`` / ``unscale_action`` / replay-buffer
    storage / critic Q forwards) all "just work" without an action-axis
    mismatch.

    Returns:
        Tuple ``(action_squashed, log_prob, aux)``. ``action_squashed`` has
        shape ``(B, 1)`` with values in ``[-1, 1]``; SB3's unscale_action
        maps that to ``[exposure_low, exposure_high]`` for the env step.
        ``log_prob`` has shape ``(B,)``. ``aux`` is a dict with keys
        ``mu`` (B, latent_dim), ``logvar`` (B, latent_dim), and ``gates``
        (B, 5) describing the LATENT z (these drive the auxiliary loss).

    Args:
        dims: :class:`SIADims` describing the observation slicing.
        latent_dim: Width of the KL-regularised latent z.
        actor_hidden: Hidden-layer sizes for gate_net, mu_net, logvar_net,
            action_mu_net, and action_log_std_net.
        exposure_high: Upper bound for the env's exposure. Kept on the
            actor as metadata only; the actor's output is ALWAYS in
            ``[-1, 1]`` and is unscaled to ``[0, exposure_high]`` by the
            SACSIA policy. Defaults to 1.5.
    """

    def __init__(
        self,
        dims: SIADims,
        latent_dim: int = 16,
        actor_hidden: Sequence[int] = (64, 64),
        exposure_high: float = 1.5,
        sparse_gates: bool = True,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if exposure_high <= 0:
            raise ValueError("exposure_high must be positive")
        self.dims = dims
        self.latent_dim = int(latent_dim)
        self.exposure_high = float(exposure_high)
        self._actor_hidden = tuple(int(h) for h in actor_hidden)
        self._sparse_gates = bool(sparse_gates)

        # Macro projection: M -> macro_small_dim.
        self.macro_proj = nn.Linear(int(dims.macro), int(dims.macro_small_dim))
        # Gate network: one sigmoid logit per gateable block (5 blocks total:
        # dispersion, pred_vol, eff_N, macro, l1_uncertainty). The trailing
        # regime_label (if present) is a group id, not a feature; we strip
        # it from the gate_net input so the gate logits are unaffected by
        # the discrete cluster id.
        gate_in_dim = int(dims.total) - int(dims.regime_label)
        self._gate_in_dim = int(gate_in_dim)
        self.gate_net = MLP(
            in_dim=int(gate_in_dim),
            hidden=self._actor_hidden,
            out_dim=5,
        )
        # Actor input: 1 (disp) + 2 (wrapper) + macro_small + 4 (risk)
        # + 1 (l1u, always 1 because zero-padded when absent).
        actor_in_dim = 1 + 2 + int(dims.macro_small_dim) + 4 + 1
        self._actor_in_dim = actor_in_dim
        # Latent z encoder (the IB bottleneck).
        self.mu_net = MLP(
            in_dim=actor_in_dim, hidden=self._actor_hidden,
            out_dim=int(latent_dim),
        )
        self.logvar_net = MLP(
            in_dim=actor_in_dim, hidden=self._actor_hidden,
            out_dim=int(latent_dim),
        )
        # 1-D action head on top of z (SB3 squashed Gaussian).
        self.action_mu_net = MLP(
            in_dim=int(latent_dim), hidden=self._actor_hidden, out_dim=1,
        )
        self.action_log_std_net = MLP(
            in_dim=int(latent_dim), hidden=self._actor_hidden, out_dim=1,
        )

    @property
    def actor_in_dim(self) -> int:
        return int(self._actor_in_dim)

    def encode(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the input-side pipeline up to ``(mu, logvar, gates, actor_in)``.

        Used by :func:`forward` and exposed separately so the test suite can
        inspect the deterministic part of the actor without sampling z.
        """
        blocks = _split_obs(obs, self.dims)
        # Strip the trailing regime_label (if any) before the gate forward;
        # the cluster id is a group label, not a feature. Without this
        # slice the gate logits would be driven by the discrete 0..7 dim.
        if self.dims.regime_label > 0:
            gate_input = obs[:, :-int(self.dims.regime_label)]
        else:
            gate_input = obs
        if self._sparse_gates:
            gates = torch.sigmoid(self.gate_net(gate_input))
        else:
            # Phase 4 no_s ablation: all 5 semantic blocks always pass
            # through unchanged. The downstream aux-loss surface stays
            # identical because the gate_l1 term on a constant 1.0 gate
            # tensor is a constant; the actor optimiser sees it as a
            # gradient-free additive bias and still updates mu_net /
            # logvar_net / action heads normally. gate_net parameters
            # exist but are inert; the Adam optimiser tolerates inert
            # params (zero grad -> zero update). The constant gate is
            # built on the obs tensor so it inherits dtype + device.
            gates = torch.ones(
                (obs.shape[0], 5), dtype=obs.dtype, device=obs.device,
            )
        macro_small = self.macro_proj(blocks["macro"])
        actor_in = torch.cat(
            [
                gates[:, 0:1] * blocks["dispersion"],
                gates[:, 1:3] * blocks["wrapper_stats"],
                gates[:, 3:4] * macro_small,
                blocks["risk_state"],
                gates[:, 4:5] * blocks["l1_uncertainty"],
            ],
            dim=-1,
        )
        mu = self.mu_net(actor_in)
        logvar = self.logvar_net(actor_in).clamp(_LOG_STD_MIN * 2.0, _LOG_STD_MAX * 2.0)
        return mu, logvar, gates, actor_in

    def _action_params(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map latent z to (action_mu, action_log_std) for the 1-D head."""
        action_mu = self.action_mu_net(z)
        action_log_std = self.action_log_std_net(z).clamp(
            _LOG_STD_MIN, _LOG_STD_MAX
        )
        return action_mu, action_log_std

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Sample tanh-squashed action and return log_prob + aux tensors.

        Args:
            obs: Float tensor of shape ``(B, dims.total)``.
            deterministic: When True, both the latent and the action are
                taken at their means (z = mu, u = action_mu). Used at
                inference time and inside the val-Sharpe selector.

        Returns:
            Tuple ``(action_squashed, log_prob, aux)``.

            - ``action_squashed``: (B, 1) tensor in ``[-1, 1]``.
            - ``log_prob``: (B,) tensor (SB3 squashed-Gaussian log-density
              of the realised action). When ``deterministic=True``, the
              log_prob is still computed at the deterministic action under
              the current (mu, log_std) for SAC's actor-loss formula; SB3's
              SAC uses non-deterministic actions in train(), so this code
              path is only exercised by callers that pass deterministic=True
              for inspection.
            - ``aux``: dict with ``mu`` (B, latent_dim), ``logvar`` (B,
              latent_dim), and ``gates`` (B, 5) describing the LATENT z.
        """
        mu, logvar, gates, _ = self.encode(obs)
        if deterministic:
            z = mu
        else:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        action_mu, action_log_std = self._action_params(z)
        action_std = torch.exp(action_log_std)
        if deterministic:
            u = action_mu
        else:
            u = action_mu + action_std * torch.randn_like(action_mu)
        action_squashed = torch.tanh(u)
        log_prob = self._squashed_log_prob(u, action_mu, action_log_std)
        aux = {"mu": mu, "logvar": logvar, "gates": gates}
        return action_squashed, log_prob, aux

    @staticmethod
    def _squashed_log_prob(
        u: torch.Tensor,
        action_mu: torch.Tensor,
        action_log_std: torch.Tensor,
    ) -> torch.Tensor:
        """Standard SB3 tanh squashed-Gaussian log_prob.

        log p(a) = log N(u | mu, sigma^2) - sum_i log(1 - tanh(u_i)^2 + eps)

        Inputs are (B, 1); output is (B,).
        """
        log_2pi = float(np.log(2.0 * np.pi))
        action_var = torch.exp(2.0 * action_log_std)
        # log N(u | mu, sigma^2), summed over action dim.
        gaussian_log_prob = -0.5 * (
            ((u - action_mu) ** 2) / action_var
            + 2.0 * action_log_std
            + log_2pi
        )
        # Squash correction: log|d tanh(u)/du| = log(1 - tanh(u)^2).
        # SB3 writes this as log(1 - a^2 + eps); we use the equivalent
        # numerically more stable formulation 2 * (log(2) - u - softplus(-2u)).
        # Either form is fine; we keep the SB3 surface form for parity.
        a = torch.tanh(u)
        squash_correction = torch.log(1.0 - a.pow(2) + _LOG_PROB_EPS)
        return (gaussian_log_prob - squash_correction).sum(dim=-1)
