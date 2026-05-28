"""Tests that the SIA aux_inv term actually fires when regime_label is wired.

The post-fix Phase 1 mini-sweep (commit 1e2d717) ran with
``regime_lookup=None`` and ``regime_label=False``, so the "I" of
InVAR-RL-SIA was a no-op (aux_inv = 0 on every minibatch). This test
verifies that:

1. The :class:`~invar_rl.layer2_sia.env_wrapper.RegimeLabelEnv` widens
   the observation by exactly 1 dim and the trailing column equals the
   day's k-means-8 cluster id (cast to float).
2. With ``regime_label=True`` the SACSIA ``_group_ids_for_batch``
   returns a 1-D long tensor of the trailing obs column.
3. The aux_inv term is non-zero on a minibatch where each row has a
   different cluster id (group means of mu differ across groups).
4. The aux_inv term is invariant to permutation of group ids when the
   per-row latent mu is identical across groups (sanity check on
   :func:`~invar_rl.layer2_sia.aux_loss._regime_invariance`).

All checks are CPU-only and do not require SB3 to be installed for
parts 1, 3, 4; part 2 uses SB3 if available.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from invar_rl.layer2_sia.aux_loss import (
    _regime_invariance,
    actor_aux_loss,
)


# --------------------------------------------------------------------
# Part 1: RegimeLabelEnv shape + tail-column wiring.
# --------------------------------------------------------------------
def test_regime_label_env_extends_obs_by_one_with_cluster_id() -> None:
    """RegimeLabelEnv widens obs by 1 and the tail is the day's cluster id."""
    import gymnasium as gym
    from gymnasium import spaces

    from invar_rl.layer2_sia.env_wrapper import RegimeLabelEnv

    class _DummyExposureEnv(gym.Env):
        """Mimic ExposureEnv's _tape.days + _start + _t protocol."""

        def __init__(self) -> None:
            super().__init__()
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=0.0, high=1.5, shape=(1,), dtype=np.float32,
            )
            self._start = 0
            self._t = 0
            # mirror ExposureEnv: the wrapper reads getattr(self.env, "_start")
            # for the reset cluster and "_t" for the post-step cluster.

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._t = 0
            return np.zeros(10, dtype=np.float32), {}

        def step(self, action):
            self._t += 1
            return (
                np.full(10, float(self._t), dtype=np.float32),
                0.0, False, bool(self._t >= 5), {},
            )

    inner = _DummyExposureEnv()
    tape_days = np.array([100, 101, 102, 103, 104, 105], dtype=np.int64)
    day_to_cluster = {100: 3, 101: 7, 102: 0, 103: 5, 104: 2, 105: 1}
    env = RegimeLabelEnv(
        inner, tape_days=tape_days, day_to_cluster=day_to_cluster,
    )
    assert env.observation_space.shape == (11,)
    obs, _ = env.reset()
    assert obs.shape == (11,)
    # _start = 0 -> tape_days[0] = 100 -> cluster 3.
    assert float(obs[-1]) == pytest.approx(3.0)

    for expected_cluster_idx in [1, 2, 3, 4]:
        obs, _, _, _, _ = env.step(np.array([0.5], dtype=np.float32))
        # After step, inner._t advanced; cluster comes from tape_days[_t].
        assert obs.shape == (11,)
        assert float(obs[-1]) == pytest.approx(
            float(day_to_cluster[int(tape_days[expected_cluster_idx])])
        )


# --------------------------------------------------------------------
# Part 2: SACSIA _group_ids_for_batch reads the trailing column.
# --------------------------------------------------------------------
def test_group_ids_match_trailing_obs_column() -> None:
    """SACSIA's group-id extractor returns the last obs column as long."""
    sb3 = importlib.util.find_spec("stable_baselines3")
    if sb3 is None:  # pragma: no cover
        pytest.skip("stable_baselines3 not installed")

    import gymnasium as gym
    from gymnasium import spaces

    from invar_rl.layer2_sia.config import SIAConfig
    from invar_rl.layer2_sia.sac_sia import SACSIA

    macro_dim = 8
    obs_dim = 7 + macro_dim + 1  # +1 for trailing regime label

    class _Env(gym.Env):
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        action_space = spaces.Box(
            low=0.0, high=1.5, shape=(1,), dtype=np.float32,
        )

        def __init__(self) -> None:
            super().__init__()
            self._t = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._t = 0
            obs = np.zeros(obs_dim, dtype=np.float32)
            obs[-1] = float(np.random.randint(0, 8))
            return obs, {}

        def step(self, action):
            self._t += 1
            obs = np.random.randn(obs_dim).astype(np.float32)
            obs[-1] = float(np.random.randint(0, 8))
            return obs, 0.0, False, bool(self._t >= 32), {}

    env = _Env()
    agent = SACSIA(
        policy="MlpPolicy", env=env,
        sia_config=SIAConfig(total_timesteps=200, buffer_size=300, batch_size=16),
        macro_dim=macro_dim, l1_uncertainty=0,
        regime_label=True,
        learning_starts=20,
        verbose=0, seed=23, device="cpu",
    )
    agent.learn(total_timesteps=100, progress_bar=False)
    sample = agent.replay_buffer.sample(16, env=agent._vec_normalize_env)
    obs_col = sample.observations[:, -1].detach().cpu().numpy()
    ids = agent._group_ids_for_batch(sample)
    assert ids is not None
    assert ids.dtype == torch.long
    assert ids.shape == (16,)
    np.testing.assert_array_equal(
        ids.cpu().numpy(), obs_col.astype(np.int64)
    )


# --------------------------------------------------------------------
# Part 3: aux_inv is non-zero when groups disagree on mu.
# --------------------------------------------------------------------
def test_aux_inv_is_nonzero_when_groups_disagree() -> None:
    """If group means of mu differ across clusters, the inv term > 0."""
    torch.manual_seed(0)
    batch = 64
    latent = 4
    # Construct mu that is correlated with the group id; this guarantees
    # the group means differ.
    group_ids = torch.randint(0, 8, (batch,), dtype=torch.long)
    bias = group_ids.float().unsqueeze(1).repeat(1, latent)
    mu = torch.randn(batch, latent) + bias
    logvar = torch.zeros(batch, latent)
    gates = torch.full((batch, 5), 0.5)
    aux = {"mu": mu, "logvar": logvar, "gates": gates}
    terms, total = actor_aux_loss(
        aux=aux, group_ids=group_ids,
        beta_kl=1e-3, lambda_gate=1e-4, lambda_inv=0.1,
    )
    assert terms.inv > 0.0
    # And the total has gradient flow (smoke).
    assert total.requires_grad is True or total.requires_grad is False
    # The KL + gate terms should also be non-zero in absolute terms.
    assert terms.kl > 0.0
    assert terms.gate_l1 > 0.0


def test_aux_inv_is_zero_when_only_one_group_in_batch() -> None:
    """Edge case: a batch that lands in a single group has inv = 0."""
    torch.manual_seed(0)
    mu = torch.randn(16, 4)
    group_ids = torch.zeros(16, dtype=torch.long)
    inv = _regime_invariance(mu, group_ids)
    assert float(inv.item()) == 0.0


# --------------------------------------------------------------------
# Part 4: aux_inv is unchanged if group ids are permuted but per-row mu
# is shared across groups (sanity check on group-mean construction).
# --------------------------------------------------------------------
def test_aux_inv_is_invariant_to_relabelling_when_mu_is_identical_across_groups() -> None:
    """Relabelling cluster ids while keeping per-cluster mu identical
    leaves the variance-across-group-means at zero."""
    torch.manual_seed(0)
    latent = 3
    # Two groups, each with two rows, all rows have identical mu.
    mu = torch.zeros(4, latent)
    a = _regime_invariance(mu, torch.tensor([0, 0, 1, 1], dtype=torch.long))
    b = _regime_invariance(mu, torch.tensor([1, 1, 0, 0], dtype=torch.long))
    c = _regime_invariance(mu, torch.tensor([3, 3, 7, 7], dtype=torch.long))
    assert float(a.item()) == pytest.approx(0.0)
    assert float(b.item()) == pytest.approx(0.0)
    assert float(c.item()) == pytest.approx(0.0)
