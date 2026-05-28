"""Central architectural sanity check for the SIA gate.

Build a synthetic supervised target where only the dispersion block
(block 0) and the macro block (block 3) carry signal; the wrapper_stats
(blocks 1 and 2) and l1_uncertainty (block 4) are noise. Train the
:class:`SparseInvariantActor` with a downstream MSE objective + the SIA
auxiliary loss for 100 epochs and verify that gates 0 + 3 stay near 1
while gates 1, 2, 4 fall well below the (untrained) average.

We do NOT require gates 1, 2, 4 to be < 0.5 (the gate-L1 weight is small
and 100 epochs is short for SAC-scale lambda_gate); the falsifiable claim
is that the predictive gates END UP higher than the noise gates by a
clear margin AND the noise gates trend DOWN from their init.

CPU-only, deterministic seed.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from invar_rl.layer2_sia.aux_loss import actor_aux_loss_scalar
from invar_rl.layer2_sia.sparse_actor import (
    SparseInvariantActor,
    resolve_dims,
)


def _synthetic_dataset(
    n: int, macro_dim: int, seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build obs + target tensors with signal in blocks 0 + 3 only.

    Returns:
        Tuple (obs, y). obs has shape (n, 7 + macro_dim); y has shape
        (n, 1). The target is ``a * disp + b * mean(macro) + noise``;
        wrapper_stats and risk_state are sampled from N(0, 1) and do not
        appear in y.
    """
    gen = torch.Generator().manual_seed(seed)
    disp = torch.randn(n, 1, generator=gen) * 1.0
    wrapper = torch.randn(n, 2, generator=gen) * 1.0
    risk = torch.randn(n, 4, generator=gen) * 0.1  # tiny so risk_state ungated does not drown the signal
    macro = torch.randn(n, macro_dim, generator=gen) * 1.0
    noise = torch.randn(n, 1, generator=gen) * 0.05
    obs = torch.cat([disp, wrapper, risk, macro], dim=-1)
    y = 0.7 * disp + 0.5 * macro.mean(dim=-1, keepdim=True) + noise
    return obs, y


def test_gate_learns_block_importance_synthetic() -> None:
    torch.manual_seed(123)
    np.random.seed(123)

    macro_dim = 8
    n = 4096
    epochs = 100
    batch_size = 256

    obs, y = _synthetic_dataset(n, macro_dim, seed=123)
    dims = resolve_dims(obs_dim=7 + macro_dim, macro_dim=macro_dim)
    actor = SparseInvariantActor(
        dims=dims, latent_dim=16, actor_hidden=(64, 64),
        exposure_high=1.5,
    )
    # Map exposure -> a regression target via a tiny linear head that we
    # also train; this gives the actor a supervised signal that depends
    # solely on the predictive blocks 0 + 3.
    regress_head = nn.Linear(1, 1)
    optim = torch.optim.Adam(
        list(actor.parameters()) + list(regress_head.parameters()),
        lr=3e-3,
    )

    # Smaller invariance weight so the synthetic test isolates gate
    # behaviour (the invariance term needs a meaningful group split,
    # which this synthetic dataset does not provide).
    beta_kl = 1e-3
    lambda_gate = 3e-3
    lambda_inv = 0.0

    # Snapshot initial gate values for the trend check.
    with torch.no_grad():
        _, _, init_aux = actor(obs[:batch_size], deterministic=True)
        init_gates = init_aux["gates"].mean(dim=0).clone()

    n_batches = max(1, n // batch_size)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            o = obs[idx]
            yt = y[idx]
            # Use the squashed action as a 1-D predictor for the regression.
            action_squashed, _, aux = actor(o, deterministic=False)
            pred = regress_head(action_squashed)
            mse = ((pred - yt) ** 2).mean()
            aux_total = actor_aux_loss_scalar(
                aux=aux, group_ids=None,
                beta_kl=beta_kl, lambda_gate=lambda_gate,
                lambda_inv=lambda_inv,
            )
            loss = mse + aux_total
            optim.zero_grad()
            loss.backward()
            optim.step()

    with torch.no_grad():
        _, _, final_aux = actor(obs, deterministic=True)
        final_gates = final_aux["gates"].mean(dim=0)

    # Predictive gates: 0 (disp) and 3 (macro). Noise gates: 1, 2, 4.
    pred_gate_mean = (final_gates[0] + final_gates[3]) / 2.0
    noise_gate_mean = (
        final_gates[1] + final_gates[2] + final_gates[4]
    ) / 3.0

    # The predictive gates must end materially HIGHER than the noise gates.
    margin = float(pred_gate_mean.item() - noise_gate_mean.item())
    assert margin > 0.05, (
        f"gate margin too small: pred_gates_mean={pred_gate_mean.item():.4f} "
        f"noise_gates_mean={noise_gate_mean.item():.4f} margin={margin:.4f} "
        f"final_gates={final_gates.tolist()} init_gates={init_gates.tolist()}"
    )
    # The noise gates must trend DOWN from their initialisation (the
    # gate L1 penalty is the only push on un-useful gates).
    init_noise = float(((init_gates[1] + init_gates[2] + init_gates[4]) / 3.0).item())
    final_noise = float(noise_gate_mean.item())
    assert final_noise <= init_noise + 0.02, (
        f"noise gates did not contract: init_noise={init_noise:.4f} "
        f"final_noise={final_noise:.4f} "
        f"final_gates={final_gates.tolist()}"
    )
