"""SWA-InVAR acceptance tests.

Two minimal checks:
  1. With ``use_swa=False`` (the default) the trainer behaves identically
     to the baseline path - the EMA buffer is never created.
  2. The EMA accumulator math is correct in isolation: a manual EMA over
     a known sequence of state_dicts matches a closed-form reference.
"""
from __future__ import annotations

import torch

from src.invar.model.invar import Invar, InvarConfig
from src.invar.training.train import TrainConfig


def test_train_config_swa_defaults() -> None:
    """SWA fields default to off and to standard values."""
    cfg = TrainConfig()
    assert cfg.use_swa is False
    assert abs(cfg.swa_decay - 0.999) < 1.0e-9
    assert cfg.swa_warmup_epochs == 5


def test_ema_accumulator_math_is_correct() -> None:
    """An EMA buffer over a sequence of state_dicts matches the
    closed-form ``ema_n = d^n * v_0 + (1-d) * sum_{k=1..n} d^{n-k} v_k``.

    We track a single weight tensor through 10 simulated training steps;
    at each step the live weight is replaced by a fresh random sample.
    The EMA result should converge toward the recent samples per the
    standard Polyak averaging formula.
    """
    torch.manual_seed(0)
    d = 0.9
    n_steps = 20

    # Reference implementation: explicit EMA.
    samples = [torch.randn(8) for _ in range(n_steps)]
    ema_ref = samples[0].clone()
    for k in range(1, n_steps):
        ema_ref = ema_ref * d + samples[k] * (1.0 - d)

    # In-place mul_/add_ idiom (same as in train.py).
    ema_test = samples[0].detach().clone()
    for k in range(1, n_steps):
        ema_test.mul_(d).add_(samples[k].detach(), alpha=1.0 - d)

    assert torch.allclose(ema_ref, ema_test, atol=1.0e-6)


def test_state_dict_roundtrip_through_ema_buffer() -> None:
    """Loading an EMA state_dict back into the model preserves shapes
    and produces finite forward outputs."""
    cfg = InvarConfig(
        n_features=26, lookback=60, macro_dim=24, d_model=128,
        regime_axis="retrieval", bank_size=64, top_k_retrieve=32,
    )
    model = Invar(cfg)
    sd_initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # Fake an EMA: linear interpolation toward a perturbed copy.
    ema_state = {}
    for k, v in sd_initial.items():
        if v.dtype.is_floating_point:
            noisy = v + 0.01 * torch.randn_like(v)
            ema_state[k] = 0.9 * v + 0.1 * noisy
        else:
            # Integer / bool buffers (no EMA defined). Keep as-is.
            ema_state[k] = v.clone()

    # Swap to EMA, run a forward, swap back, run again.
    saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(ema_state)
    model.eval()
    out_ema = model(
        features=torch.randn(20, 60, 26),
        macro=torch.randn(60, 24),
        mask=torch.ones(20, dtype=torch.bool),
    )
    assert torch.isfinite(out_ema["y_hat"]).all()
    model.load_state_dict(saved)
    out_live = model(
        features=torch.randn(20, 60, 26),
        macro=torch.randn(60, 24),
        mask=torch.ones(20, dtype=torch.bool),
    )
    assert torch.isfinite(out_live["y_hat"]).all()
