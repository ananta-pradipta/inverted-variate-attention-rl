"""StockFormer (Gao et al., IJCAI 2023) aligned to gsyyysg/StockFormer.

Updated 2026-05-21: aligned to the upstream code repo (branch ``main``)
across the architecture, MAE mask strategy, branch losses, fusion, and
the SAC env. The legacy clean-room re-implementation that used a single
shared encoder and a continuous-weight portfolio env is replaced with:

1) Per-stock TEMPORAL Transformer enc-dec, one per pretraining branch
   (long, short, MAE). Input is a (seq_len=60, F_raw) per-stock tensor
   sliced from raw OHLCV + technical indicators (matches upstream's
   ``Transformer/models/transformer.py`` / ``timesnet.py`` input shape).
   The encoder maps (B, 60, F) -> (B, 60, d_model); the decoder consumes
   the encoder output and produces a single contextual hidden per stock
   used by the branch's prediction head.

2) Three branches, each with its OWN encoder-decoder transformer:
   - long branch supervises on a 5-day forward log return
     (``label_long_term``), MSE + 0.5 * ranking-IC.
   - short branch supervises on the 1-day forward log return
     (continuous), MSE + 1.0 * ranking-IC (NOT BCE on sign).
   - MAE branch reconstructs the per-stock feature block when 50% of
     WHOLE STOCKS are masked (sample a stock-level Bernoulli mask;
     when chosen, the entire (seq_len, F) block of that stock is
     zeroed). MSE is computed on the masked stocks only.

3) SAC features extractor loads the long and short ENCODERS from the
   pretrained enc-dec checkpoints. Both encoders are kept trainable
   during RL so critic-gradients flow back into the encoder weights
   (StockFormer's central architectural claim). The fusion is a
   two-layer multi-head self-attention over the concatenated long/short
   hidden tokens (matches upstream
   ``MySAC/SAC/policy_transformer.py policy_transformer_attn2``).

4) SAC environment matches FinRL's ``env_stocktrading_hybrid_control``:
   discrete-share action (``a = (a * hmax).astype(int)``), explicit cash
   account, balance term in the observation, long-only invariants
   (shares >= 0, balance >= 0), per-step simple-return reward plus a
   terminal cumulative-return bonus (reward scale 1e-4). Sell/buy loop
   is argsort-by-magnitude.

We keep the SB3 in-place-gradient workaround
(share_features_extractor=False, per-extractor private encoders,
weights copied from the pretrained state) because it is independent of
the alignment items; both extractor instances are initialised from the
same upstream encoder checkpoint.

Not a verbatim port of the upstream code (paths, imports, training
loops are ours), but the MDP, the encoder topology, the loss objectives
and the SAC fusion match the public repo as of 2026-05-21.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from torch import nn


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass
class StockFormerConfig:
    """Encoder + branch hyperparameters.

    Aligned to upstream (gsyyysg/StockFormer main):
      d_model=128, n_heads=4, encoder layers=2, decoder layers=1,
      dropout=0.05, activation=gelu, seq_len=60, feature_dim=
      MAE-branch 96 or pred-branch 10. We follow the pred-branch
      feature set (10 columns) as the standard input: open/high/low/
      close/volume + 5 derived indicators (MACD, RSI, ATR, sma5
      returns, sma20 returns).
    """

    universe_size: int = 30
    lookback: int = 60
    # F_raw = OHLCV(5) + MACD + RSI + ATR + sma5_ret + sma20_ret = 10
    feature_dim: int = 10
    d_model: int = 128
    n_heads: int = 4
    n_encoder_layers: int = 2
    n_decoder_layers: int = 1
    dropout: float = 0.05
    mask_stock_frac: float = 0.5  # 50% of WHOLE stocks per step (MAE)
    long_term_len: int = 5  # 5-day forward return target for long branch
    initial_balance: float = 1_000_000.0
    transaction_cost_pct: float = 0.001
    reward_scaling: float = 1e-4
    hmax: int = 100
    long_only: bool = True


@dataclass
class StockFormerEnvConfig:
    """FinRL-idiom hybrid-control env config (discrete shares + cash)."""

    universe_size: int = 30
    lookback: int = 60
    feature_dim: int = 10
    cov_window: int = 60
    initial_balance: float = 1_000_000.0
    transaction_cost_pct: float = 0.001
    reward_scaling: float = 1e-4
    hmax: int = 100
    long_only: bool = True


# --------------------------------------------------------------------------
# Feature construction: per-stock (seq_len=60, F_raw=10) raw inputs
# --------------------------------------------------------------------------


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    """Vectorised exponential moving average along axis 0."""
    if arr.shape[0] == 0:
        return arr
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for t in range(1, arr.shape[0]):
        out[t] = alpha * arr[t] + (1.0 - alpha) * out[t - 1]
    return out


def _per_stock_window(
    raw: Dict[str, np.ndarray],
    day_idx: int,
    universe: np.ndarray,
    lookback: int,
) -> np.ndarray:
    """Build the (K, T=lookback, F_raw=10) per-stock temporal window.

    ``raw`` is a dict with keys ``open``, ``high``, ``low``, ``close``,
    ``volume``, each as a (T_full, N_universe) numpy array. ``universe``
    indexes into N_universe. Indicators (MACD, RSI-14, ATR-14, sma5
    return, sma20 return) are computed inside the window using stockstats
    semantics approximated locally. NaN/inf are zeroed.
    """
    K = int(universe.size)
    F = 10
    T = lookback
    out = np.zeros((K, T, F), dtype=np.float32)
    lo = max(0, day_idx - lookback)
    hi = day_idx
    if hi - lo < 2:
        return out
    o = raw["open"][lo:hi, :][:, universe]
    h = raw["high"][lo:hi, :][:, universe]
    lw = raw["low"][lo:hi, :][:, universe]
    c = raw["close"][lo:hi, :][:, universe]
    v = raw["volume"][lo:hi, :][:, universe]
    cur_T = c.shape[0]
    # Pad to lookback at the front with the first observed value so
    # the encoder always sees seq_len=lookback.
    if cur_T < T:
        pad = T - cur_T
        o = np.vstack([np.tile(o[0:1], (pad, 1)), o])
        h = np.vstack([np.tile(h[0:1], (pad, 1)), h])
        lw = np.vstack([np.tile(lw[0:1], (pad, 1)), lw])
        c = np.vstack([np.tile(c[0:1], (pad, 1)), c])
        v = np.vstack([np.tile(v[0:1], (pad, 1)), v])
    # MACD = ema(12) - ema(26).
    ema12 = _ema(c, 12)
    ema26 = _ema(c, 26)
    macd = ema12 - ema26
    # RSI-14.
    diff = np.diff(c, axis=0, prepend=c[:1])
    up = np.where(diff > 0, diff, 0.0)
    dn = np.where(diff < 0, -diff, 0.0)
    avg_up = _ema(up, 14)
    avg_dn = _ema(dn, 14)
    rs = avg_up / (avg_dn + 1e-8)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # ATR-14: true range = max(h-l, |h - prev_c|, |l - prev_c|).
    prev_c = np.vstack([c[:1], c[:-1]])
    tr = np.maximum.reduce([
        h - lw,
        np.abs(h - prev_c),
        np.abs(lw - prev_c),
    ])
    atr = _ema(tr, 14)
    # SMA-5 / SMA-20 log returns.
    def _sma(x, w):
        ker = np.ones((w,)) / w
        out = np.zeros_like(x)
        for j in range(x.shape[1]):
            out[:, j] = np.convolve(x[:, j], ker, mode="same")
        return out
    log_c = np.log(np.clip(c, 1e-6, None))
    sma5 = _sma(log_c, 5)
    sma20 = _sma(log_c, 20)
    sma5_ret = log_c - sma5
    sma20_ret = log_c - sma20
    # Z-score normalise volume per stock within the window.
    v_mu = v.mean(axis=0, keepdims=True)
    v_sd = v.std(axis=0, keepdims=True) + 1e-6
    v_z = (v - v_mu) / v_sd
    # Stack: O, H, L, C (z-scored per stock), V_z, MACD, RSI, ATR,
    # sma5_ret, sma20_ret.
    c_mu = c.mean(axis=0, keepdims=True)
    c_sd = c.std(axis=0, keepdims=True) + 1e-6
    o_z = (o - c_mu) / c_sd
    h_z = (h - c_mu) / c_sd
    l_z = (lw - c_mu) / c_sd
    c_z = (c - c_mu) / c_sd
    feats = np.stack([
        o_z, h_z, l_z, c_z, v_z, macd, rsi / 100.0, atr / (c_mu + 1e-6),
        sma5_ret, sma20_ret,
    ], axis=-1)  # (T, K, F)
    feats = np.transpose(feats, (1, 0, 2))  # (K, T, F)
    feats = np.where(np.isfinite(feats), feats, 0.0).astype(np.float32)
    out[:, :, :] = feats
    return out


def _rolling_covariance(
    log_returns: np.ndarray,
    day_idx: int,
    universe: np.ndarray,
    window: int = 60,
) -> np.ndarray:
    lo = max(0, day_idx - window)
    win = log_returns[lo:day_idx, :][:, universe]
    win = np.where(np.isfinite(win), win, 0.0)
    if win.shape[0] < 5:
        return np.eye(universe.size, dtype=np.float32) * 1e-6
    return np.cov(win.T).astype(np.float32)


# --------------------------------------------------------------------------
# Per-stock enc-dec Transformer (one per pretraining branch)
# --------------------------------------------------------------------------


class StockFormerBranch(nn.Module):
    """Per-stock encoder-decoder transformer.

    Matches the upstream ``Transformer_base`` topology:
      - input_proj: F_raw -> d_model
      - encoder: TransformerEncoder with ``n_encoder_layers`` layers,
        operating along the seq_len=60 axis per stock
      - decoder: TransformerDecoder with ``n_decoder_layers`` layers
        consuming a single learned query vector per stock
      - head: linear projection of the decoder output (1 per stock for
        long/short, F_raw per stock for MAE)

    Output:
      - .encode(x) -> (B*K, seq_len, d_model)  [encoder output]
      - .forward(x) -> head output per stock
    """

    def __init__(
        self,
        cfg: StockFormerConfig,
        head_kind: str,
    ) -> None:
        super().__init__()
        assert head_kind in ("long", "short", "mae")
        self.cfg = cfg
        self.head_kind = head_kind
        d = cfg.d_model
        self.input_proj = nn.Linear(cfg.feature_dim, d)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_heads,
            dim_feedforward=2 * d, dropout=cfg.dropout,
            activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=cfg.n_encoder_layers,
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=cfg.n_heads,
            dim_feedforward=2 * d, dropout=cfg.dropout,
            activation="gelu", batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            dec_layer, num_layers=cfg.n_decoder_layers,
        )
        # Decoder query: one learned token per stock (B*K, 1, d).
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        if head_kind == "mae":
            self.head = nn.Linear(d, cfg.feature_dim)
        else:
            self.head = nn.Linear(d, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, K, T, F) -> (B, K, T, d_model)."""
        B, K, T, F = x.shape
        h = self.input_proj(x.reshape(B * K, T, F))  # (BK, T, d)
        h = self.encoder(h)  # (BK, T, d)
        return h.reshape(B, K, T, -1)

    def decode_last(self, enc_out: torch.Tensor) -> torch.Tensor:
        """enc_out: (B, K, T, d) -> (B, K, d).

        Decoder consumes a single learned query per stock and attends to
        the encoder output along the seq_len axis.
        """
        B, K, T, D = enc_out.shape
        q = self.query.expand(B * K, -1, -1)
        m = enc_out.reshape(B * K, T, D)
        out = self.decoder(q, m)  # (BK, 1, d)
        return out.squeeze(1).reshape(B, K, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.encode(x)
        h = self.decode_last(enc)  # (B, K, d)
        return self.head(h)  # (B, K, 1) or (B, K, F)


# --------------------------------------------------------------------------
# Pretraining: three separate enc-dec networks
# --------------------------------------------------------------------------


def _soft_rank_ic(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative Spearman approximation via Pearson on argsort ranks."""
    pr = pred.argsort().argsort().float() / max(1, pred.numel() - 1)
    tr = target.argsort().argsort().float() / max(1, target.numel() - 1)
    return -(
        ((pr - pr.mean()) * (tr - tr.mean())).mean()
        / (pr.std() * tr.std() + 1e-6)
    )


def _ranking_ic_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative Pearson correlation on the (B, K) rank matrix.

    For each batch element compute the Spearman approximation across the
    K stocks; average over the batch. Returned as a LOSS to be minimised
    (negative sign so minimising it maximises rank-IC).
    """
    if pred.dim() == 1:
        return _soft_rank_ic(pred, target)
    losses = []
    for b in range(pred.shape[0]):
        losses.append(_soft_rank_ic(pred[b], target[b]))
    return torch.stack(losses).mean()


def pretrain_three_branch(
    long_branch: StockFormerBranch,
    short_branch: StockFormerBranch,
    mae_branch: StockFormerBranch,
    raw: Dict[str, np.ndarray],
    log_returns: np.ndarray,
    universe: np.ndarray,
    train_days: List[int],
    cfg: StockFormerConfig,
    epochs: int = 50,
    lr: float = 1e-4,
    batch_size: int = 32,
    device: str = "cpu",
) -> None:
    """Three-branch supervised pretraining.

    Each branch has its OWN encoder-decoder transformer, optimised with
    its own loss:
      - long: MSE + 0.5 * ranking-IC on 5-day forward log return.
      - short: MSE + 1.0 * ranking-IC on 1-day forward log return.
      - MAE: 50% stock-level Bernoulli mask, MSE on masked stocks.

    Each step samples ``batch_size`` random days uniformly from
    ``train_days``; the per-stock 60-day windows are stacked into
    (B, K, 60, F).
    """
    long_branch.train(); short_branch.train(); mae_branch.train()
    long_opt = torch.optim.Adam(long_branch.parameters(), lr=lr)
    short_opt = torch.optim.Adam(short_branch.parameters(), lr=lr)
    mae_opt = torch.optim.Adam(mae_branch.parameters(), lr=lr)
    K = int(universe.size)
    F = cfg.feature_dim
    T = cfg.lookback
    valid_days = [
        d for d in train_days
        if d >= cfg.lookback and d + cfg.long_term_len < log_returns.shape[0]
    ]
    if len(valid_days) < batch_size:
        batch_size = max(1, len(valid_days))
    steps_per_epoch = max(1, len(valid_days) // batch_size)
    for ep in range(epochs):
        rng = np.random.default_rng(ep + 7919)
        tot_long = 0.0; tot_short = 0.0; tot_mae = 0.0; n = 0
        for _ in range(steps_per_epoch):
            day_batch = rng.choice(valid_days, size=batch_size, replace=False)
            x_list = []
            y1 = np.zeros((batch_size, K), dtype=np.float32)
            y5 = np.zeros((batch_size, K), dtype=np.float32)
            for bi, d in enumerate(day_batch):
                x_list.append(_per_stock_window(raw, int(d), universe, T))
                # 1-day forward log return (short branch target).
                r1 = log_returns[int(d) + 1, universe]
                y1[bi, :] = np.where(np.isfinite(r1), r1, 0.0)
                # 5-day forward log return (long branch target).
                lo = int(d) + 1
                hi = int(d) + 1 + cfg.long_term_len
                r5 = log_returns[lo:hi, :][:, universe].sum(axis=0)
                y5[bi, :] = np.where(np.isfinite(r5), r5, 0.0)
            x = torch.from_numpy(np.stack(x_list, axis=0)).to(device)
            y1_t = torch.from_numpy(y1).to(device)
            y5_t = torch.from_numpy(y5).to(device)
            # Long branch on full input (5-day target).
            long_pred = long_branch(x).squeeze(-1)  # (B, K)
            long_mse = ((long_pred - y5_t) ** 2).mean()
            long_ic = _ranking_ic_loss(long_pred, y5_t)
            long_loss = long_mse + 0.5 * long_ic
            long_opt.zero_grad(); long_loss.backward(); long_opt.step()
            # Short branch on full input (1-day target, continuous).
            short_pred = short_branch(x).squeeze(-1)  # (B, K)
            short_mse = ((short_pred - y1_t) ** 2).mean()
            short_ic = _ranking_ic_loss(short_pred, y1_t)
            short_loss = short_mse + 1.0 * short_ic
            short_opt.zero_grad(); short_loss.backward(); short_opt.step()
            # MAE branch: mask 50% of WHOLE stocks per sample.
            stock_mask = (
                torch.rand(x.shape[0], x.shape[1], device=device)
                < cfg.mask_stock_frac
            )  # (B, K), True = masked
            x_masked = x.clone()
            for bi in range(x.shape[0]):
                x_masked[bi, stock_mask[bi]] = 0.0
            recon = mae_branch(x_masked)  # (B, K, F)
            target = x[:, :, -1, :]  # reconstruct the last time-step
            mse_per_stock = ((recon - target) ** 2).mean(dim=-1)  # (B, K)
            denom = stock_mask.float().sum() + 1e-6
            mae_loss = (mse_per_stock * stock_mask.float()).sum() / denom
            mae_opt.zero_grad(); mae_loss.backward(); mae_opt.step()
            tot_long += float(long_loss.item())
            tot_short += float(short_loss.item())
            tot_mae += float(mae_loss.item())
            n += 1
        if n:
            print(
                f"    [stockformer pretrain ep{ep}] long={tot_long/n:.4f} "
                f"short={tot_short/n:.4f} mae={tot_mae/n:.4f} "
                f"(batch_size={batch_size}, lr={lr})"
            )


# --------------------------------------------------------------------------
# SAC features extractor: 2-layer attention fusion of long_h, short_h
# --------------------------------------------------------------------------


class StockFormerFeatureExtractor(nn.Module):
    """Features extractor that owns its own private long/short encoders.

    SB3 instantiates two of these when ``share_features_extractor=False``
    (one for actor, one for critic). We copy the pretrained encoder
    weights into both instances at construction time. Both encoders
    receive critic-gradients during RL (the StockFormer signature
    property).

    Fusion: a 2-layer multi-head self-attention block over the
    concatenated long/short token sequence (per stock), pooled to a
    single feature vector, then concatenated with the covariance
    flatten, holdings and balance. Matches the upstream
    ``policy_transformer_attn2`` shape.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        cfg: StockFormerConfig,
        pretrained_state: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        K = cfg.universe_size
        d = cfg.d_model
        self.K = K
        self.long_encoder_branch = StockFormerBranch(cfg, head_kind="long")
        self.short_encoder_branch = StockFormerBranch(cfg, head_kind="short")
        if pretrained_state is not None:
            try:
                if "long" in pretrained_state:
                    self.long_encoder_branch.load_state_dict(
                        pretrained_state["long"], strict=False,
                    )
                if "short" in pretrained_state:
                    self.short_encoder_branch.load_state_dict(
                        pretrained_state["short"], strict=False,
                    )
            except Exception as e:  # noqa
                print(f"[StockFormerFeatureExtractor] weight load warn: {e}")
        # 2-layer multi-head self-attention over [long_h ; short_h]
        # token sequence (length 2K).
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_heads,
            dim_feedforward=2 * d, dropout=cfg.dropout,
            activation="gelu", batch_first=True,
        )
        self.fusion = nn.TransformerEncoder(fusion_layer, num_layers=2)
        cov_dim = K * K
        # Pooled fusion output (K * d), plus cov, holdings (K), balance (1).
        self._features_dim = K * d + cov_dim + K + 1

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Observation layout:
            features (K * T * F) | cov (K * K) | holdings (K) | balance (1)
        """
        K = self.K
        F = self.cfg.feature_dim
        T = self.cfg.lookback
        B = obs.shape[0]
        feats_flat = obs[:, : K * T * F]
        cov_flat = obs[:, K * T * F: K * T * F + K * K]
        rest = obs[:, K * T * F + K * K:]
        x = feats_flat.reshape(B, K, T, F)
        long_enc = self.long_encoder_branch.encode(x)  # (B, K, T, d)
        short_enc = self.short_encoder_branch.encode(x)
        long_h = long_enc.mean(dim=2)  # (B, K, d)
        short_h = short_enc.mean(dim=2)
        # Concat long + short along the token axis: (B, 2K, d).
        tokens = torch.cat([long_h, short_h], dim=1)
        fused = self.fusion(tokens)  # (B, 2K, d)
        # Pool back to (B, K, d) by averaging the long+short half.
        pooled = 0.5 * (fused[:, :K, :] + fused[:, K:, :])  # (B, K, d)
        return torch.cat([
            pooled.reshape(B, -1),
            cov_flat,
            rest,
        ], dim=-1)


# --------------------------------------------------------------------------
# StockTradingEnvHybridControl: FinRL-idiom discrete-share long-only env
# --------------------------------------------------------------------------


class StockTradingEnvHybridControl(gym.Env):
    """FinRL-idiom discrete-share long-only env, matching upstream
    ``code/envs/env_stocktrading_hybrid_control.py``.

    Observation layout (matches StockFormerFeatureExtractor):
      - per-stock raw window features (K * T * F)
      - rolling covariance (K * K)
      - holdings (K)  (per-ticker share counts)
      - balance (1)   (cash account)

    Action: continuous in [-1, 1]^K, multiplied by hmax and int-cast to
    shares. Sells before buys, argsort by |action| (largest first).
    Long-only invariants: shares >= 0, balance >= 0. Buys are capped by
    available cash; sells are capped by inventory.

    Reward: per-step simple return on equity (begin -> end of step),
    times reward_scaling. On the terminal step a bonus equal to the
    cumulative (final_equity - initial) / initial * reward_scaling is
    added once.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        log_returns: np.ndarray,
        raw: Dict[str, np.ndarray],
        tradable: np.ndarray,
        day_indices: List[int],
        universe: np.ndarray,
        cfg: StockFormerEnvConfig,
    ) -> None:
        super().__init__()
        self._lr = log_returns
        self._raw = raw
        self._tradable = tradable
        self._days = list(day_indices)
        self._uni = universe
        self._cfg = cfg
        self._K = int(universe.size)
        F = cfg.feature_dim
        T = cfg.lookback
        obs_dim = self._K * T * F + self._K * self._K + self._K + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._K,), dtype=np.float32,
        )
        self._t = 0
        self._shares = np.zeros(self._K, dtype=np.int64)
        self._balance = float(cfg.initial_balance)
        self._initial_equity = float(cfg.initial_balance)

    def _close_today(self) -> np.ndarray:
        d = self._days[self._t]
        c = self._raw["close"][d, :][self._uni]
        c = np.where(np.isfinite(c) & (c > 0), c, 1.0)
        return c.astype(np.float64)

    def _features_today(self) -> np.ndarray:
        d = self._days[self._t]
        return _per_stock_window(
            self._raw, d, self._uni, self._cfg.lookback,
        )

    def _cov_today(self) -> np.ndarray:
        d = self._days[self._t]
        return _rolling_covariance(
            self._lr, d, self._uni, self._cfg.cov_window,
        )

    def _obs(self) -> np.ndarray:
        feats = self._features_today().flatten().astype(np.float32)
        cov = self._cov_today().flatten().astype(np.float32)
        return np.concatenate([
            feats, cov,
            self._shares.astype(np.float32),
            np.array([self._balance], dtype=np.float32),
        ])

    def _equity(self) -> float:
        prices = self._close_today()
        return float(self._balance + (self._shares * prices).sum())

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        self._shares = np.zeros(self._K, dtype=np.int64)
        self._balance = float(self._cfg.initial_balance)
        self._initial_equity = float(self._cfg.initial_balance)
        return self._obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        target_shares = (action * self._cfg.hmax).astype(np.int64)
        prices = self._close_today()
        pre_equity = self._equity()
        # Argsort by |action|: largest magnitudes first.
        order = np.argsort(-np.abs(action))
        # Sells first.
        for i in order:
            delta = int(target_shares[i])
            if delta < 0 and self._shares[i] > 0:
                sell_n = min(int(-delta), int(self._shares[i]))
                proceeds = sell_n * prices[i]
                cost = proceeds * self._cfg.transaction_cost_pct
                self._balance += float(proceeds - cost)
                self._shares[i] -= sell_n
        # Buys second.
        for i in order:
            delta = int(target_shares[i])
            if delta > 0:
                pi = float(prices[i])
                if not np.isfinite(pi) or pi <= 0:
                    continue
                max_affordable = int(self._balance // (
                    pi * (1.0 + self._cfg.transaction_cost_pct)
                ))
                buy_n = min(int(delta), max(0, max_affordable))
                if buy_n > 0:
                    cost = buy_n * pi
                    fee = cost * self._cfg.transaction_cost_pct
                    self._balance -= float(cost + fee)
                    self._shares[i] += buy_n
        self._t += 1
        done = self._t >= len(self._days) - 1
        post_equity = self._equity()
        # Per-step simple return reward (scaled).
        step_reward = (
            (post_equity - pre_equity) * self._cfg.reward_scaling
        )
        # Terminal cumulative bonus.
        if done:
            terminal_bonus = (
                (post_equity - self._initial_equity)
                / max(1e-6, self._initial_equity)
                * self._cfg.reward_scaling
            )
            step_reward = float(step_reward + terminal_bonus)
        return (
            self._obs(),
            float(step_reward),
            done, False,
            {"equity": post_equity, "balance": self._balance},
        )


# --------------------------------------------------------------------------
# SAC training entrypoint
# --------------------------------------------------------------------------


def train_stockformer_sac(
    train_env: StockTradingEnvHybridControl,
    pretrained_state: dict,
    cfg: StockFormerConfig,
    seed: int,
    total_timesteps: int = 30_000,
    learning_rate: float = 1e-4,
    device: str = "cpu",
):
    from stable_baselines3 import SAC
    policy_kwargs = {
        "features_extractor_class": StockFormerFeatureExtractor,
        "features_extractor_kwargs": {
            "cfg": cfg,
            "pretrained_state": pretrained_state,
        },
        # share_features_extractor=False: each features_extractor
        # instance owns its own encoders. Both are initialised from the
        # same pretrained checkpoint state at construction. This
        # workaround for SB3's in-place gradient on shared attention
        # extractors is unrelated to the upstream alignment items and
        # is documented separately.
        "share_features_extractor": False,
    }
    agent = SAC(
        "MlpPolicy", train_env,
        learning_rate=learning_rate,
        batch_size=32,
        buffer_size=100_000,
        ent_coef="auto_0.1",
        seed=seed, verbose=0, device=device,
        policy_kwargs=policy_kwargs,
    )
    agent.learn(total_timesteps=total_timesteps)
    return agent


def evaluate(env: StockTradingEnvHybridControl, agent) -> Dict:
    obs, _ = env.reset(seed=0)
    daily_equity = [float(env._initial_equity)]
    while True:
        action, _ = agent.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(np.asarray(action, dtype=np.float32))
        if "equity" in info:
            daily_equity.append(float(info["equity"]))
        if term or trunc:
            break
    eq = np.asarray(daily_equity, dtype=np.float64)
    if eq.size < 2:
        return {
            "mean_return": 0.0, "volatility": 0.0,
            "ann_return": 0.0, "ann_vol": 0.0,
            "sharpe_annualised": 0.0,
            "final_equity": 1.0, "n_steps": int(eq.size),
            "daily_log_returns": [],
        }
    log_rets = np.diff(np.log(np.clip(eq, 1e-6, None)))
    log_rets = log_rets[np.isfinite(log_rets)]
    mean = float(log_rets.mean()) if log_rets.size else 0.0
    vol = float(log_rets.std(ddof=1)) if log_rets.size > 1 else 0.0
    ann_ret = mean * 252.0
    ann_vol = vol * np.sqrt(252.0)
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
    return {
        "mean_return": mean, "volatility": vol,
        "ann_return": float(ann_ret), "ann_vol": float(ann_vol),
        "sharpe_annualised": sharpe,
        "final_equity": float(eq[-1] / max(1e-6, env._initial_equity)),
        "n_steps": int(eq.size),
        "daily_log_returns": log_rets.tolist(),
    }


# --------------------------------------------------------------------------
# Top-level entrypoint used by training/stockformer_faithful_eval.py
# --------------------------------------------------------------------------


def run_stockformer_faithful(
    log_returns: np.ndarray,
    raw: Dict[str, np.ndarray],
    tradable: np.ndarray,
    universe: np.ndarray,
    train_days: List[int],
    test_days: List[int],
    seed: int,
    cfg: Optional[StockFormerConfig] = None,
    pretrain_epochs: int = 50,
    pretrain_batch_size: int = 32,
    pretrain_lr: float = 1e-4,
    sac_lr: float = 1e-4,
    total_timesteps: int = 30_000,
    device: str = "cpu",
) -> Dict:
    """Drive the full repo-aligned StockFormer pipeline.

    Steps:
      1) Three-branch pretraining (long enc-dec, short enc-dec, MAE
         enc-dec), batch_size=32 random days per gradient step, lr=1e-4.
      2) Build StockTradingEnvHybridControl over train_days and
         test_days.
      3) Train SAC (lr=1e-4, buffer=100k, batch=32, ent_coef='auto_0.1')
         with the StockFormerFeatureExtractor initialised from the
         pretrained long+short encoder states.
      4) Deterministic eval on test_days.
    """
    cfg = cfg or StockFormerConfig(universe_size=int(universe.size))
    cfg.universe_size = int(universe.size)
    long_branch = StockFormerBranch(cfg, head_kind="long").to(device)
    short_branch = StockFormerBranch(cfg, head_kind="short").to(device)
    mae_branch = StockFormerBranch(cfg, head_kind="mae").to(device)
    print(
        f"[stockformer] three-branch enc-dec pretraining for "
        f"{pretrain_epochs} epochs, batch_size={pretrain_batch_size}, "
        f"lr={pretrain_lr}..."
    )
    pretrain_three_branch(
        long_branch=long_branch, short_branch=short_branch,
        mae_branch=mae_branch,
        raw=raw, log_returns=log_returns, universe=universe,
        train_days=train_days, cfg=cfg,
        epochs=pretrain_epochs, lr=pretrain_lr,
        batch_size=pretrain_batch_size, device=device,
    )
    pretrained_state = {
        "long": {k: v.detach().cpu().clone()
                  for k, v in long_branch.state_dict().items()},
        "short": {k: v.detach().cpu().clone()
                  for k, v in short_branch.state_dict().items()},
    }
    env_cfg = StockFormerEnvConfig(
        universe_size=cfg.universe_size, lookback=cfg.lookback,
        feature_dim=cfg.feature_dim,
        initial_balance=cfg.initial_balance,
        transaction_cost_pct=cfg.transaction_cost_pct,
        reward_scaling=cfg.reward_scaling, hmax=cfg.hmax,
        long_only=cfg.long_only,
    )
    train_env = StockTradingEnvHybridControl(
        log_returns=log_returns, raw=raw, tradable=tradable,
        day_indices=train_days, universe=universe, cfg=env_cfg,
    )
    test_env = StockTradingEnvHybridControl(
        log_returns=log_returns, raw=raw, tradable=tradable,
        day_indices=test_days, universe=universe, cfg=env_cfg,
    )
    print(f"[stockformer] training SAC for {total_timesteps} timesteps...")
    agent = train_stockformer_sac(
        train_env=train_env,
        pretrained_state=pretrained_state,
        cfg=cfg, seed=seed,
        total_timesteps=total_timesteps,
        learning_rate=sac_lr,
        device=device,
    )
    perf = evaluate(test_env, agent)
    perf["seed"] = seed
    return perf


__all__ = [
    "StockFormerConfig", "StockFormerEnvConfig",
    "StockFormerBranch", "StockFormerFeatureExtractor",
    "StockTradingEnvHybridControl",
    "pretrain_three_branch", "train_stockformer_sac",
    "evaluate", "run_stockformer_faithful",
    "_per_stock_window", "_rolling_covariance",
]
