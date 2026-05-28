"""DeepTrader baseline: ASU + MSU + Portfolio Generator with REINFORCE.

Native re-implementation of DeepTrader (Wang, Wang, Zheng, Wang, Yu;
"DeepTrader: A Deep Reinforcement Learning Approach for Risk-Return
Balanced Portfolio Management with Market Conditions Embedding";
AAAI 2021). Upstream reference code: ``github.com/CMACH508/DeepTrader``.

This module is rewritten (2026-05-22) to be faithful to the upstream
PyTorch implementation in ``src/model``, ``src/agent.py`` and
``src/environment/portfolio_env.py``. The previous version diverged on
six HIGH-severity items catalogued in
``drafts/invar_rl_deeptrader_audit_2026-05-22.md``; this rewrite addresses
all of them.

Architecture summary:
- ``ASU`` (Asset Scoring Unit): a 4-block adaptive GCN over the GICS
  sector adjacency. Inputs are per-stock windows of shape
  ``(B, N, T, F_asu)``. Output: per-stock sigmoid score ``s in (0, 1)``
  shape ``(B, N)``. Upstream form: ``s = sigmoid(linear(GCN(x)))``;
  short-side ranking criterion is ``sign(s) * (1 - s)``, NOT a separate
  head. We use the same low-rank-1 learnable adaptive adjacency
  ``softmax(relu(nodevec @ nodevec.T))`` from upstream.
- ``MSU`` (Market Scoring Unit): a single-layer LSTM (hidden=128) with
  Bahdanau-style attention pooling over the L hidden states, then
  ``BatchNorm1d -> Linear(hidden, 2)`` emitting ``(mu, raw_sigma)``.
  ``sigma = softplus(raw_sigma)``. Leverage is sampled from
  ``Normal(mu, sigma)`` at train time and clamped to ``[0, 1]``; the
  eval-time leverage is ``clamp(mu, 0, 1)``.
- ``DeepTraderActor``: composes ASU + MSU and emits
  ``(weights, rho, log_prob_rho, scores_p)`` where ``weights`` is shape
  ``(B, 2N)`` (long book + short book, softmaxed over top-G entries
  within each group; rest zero) and ``scores_p = softmax(scores, dim=-1)``
  is the full-distribution policy used by the upstream ASU surrogate.
- ``DeepTraderEnv``: gym-style env that rebalances every ``trade_len``
  trading days (default 21 = monthly), holding weights constant
  intra-month so they drift with returns (geometric compounding).
  Reward is the geometric 21-day return minus turnover fee.

Training: REINFORCE with two heads, batched across ``batch_size``
parallel trajectories.
- ASU surrogate (upstream form, low variance, bounded):
    ``gradient_asu_step = log(sum_n softmax(scores)_n * z_n)``
  where ``z_n`` is the cross-sectional z-score of next-period returns at
  step ``t``. Sum these steps along the trajectory axis.
- MSU surrogate: ``gradient_rho = (-2 * (mdd - 0.5)) * log_prob_rho``
  summed across steps; ``mdd`` is the per-trajectory max drawdown of
  the equity curve.
- Batch reward normalisation: per-step rewards (used for the MSU
  reward weighting and for diagnostic equity) are z-scored across the
  batch dimension before back-prop. (Upstream does this for
  ``rewards_total - market_avg_return``; we approximate
  ``market_avg_return`` with the cross-sectional mean of next-period
  returns at each step.)
- Total loss: ``-(gamma * gradient_rho + gradient_asu).mean()`` over
  the (batch, steps) axes. ``gamma = 0.05`` and ``lr = 1e-6`` are
  upstream's values.

Hyperparameters: see :class:`DeepTraderTrainConfig` defaults; they track
``src/hyper.json`` (``batch_size=37``, ``trade_len=21``, ``max_steps=12``,
``max_grad_norm=100``, ``gamma=0.05``, ``lr=1e-06``, ``G=4`` is set on
:class:`DeepTraderActor`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal
from torch.nn import functional as F


# --------------------------------------------------------------------- #
# Adjacency utilities
# --------------------------------------------------------------------- #


def symmetric_normalize(adj: torch.Tensor) -> torch.Tensor:
    """Symmetric normalisation ``D^{-1/2} (A + I) D^{-1/2}`` for a GCN.

    Args:
        adj: Square adjacency tensor of shape ``(N, N)``. Need not be
            binary, but assumed nonnegative and symmetric.

    Returns:
        Normalised tensor of shape ``(N, N)``.
    """
    n = adj.shape[0]
    a_hat = adj + torch.eye(n, dtype=adj.dtype, device=adj.device)
    deg = a_hat.sum(dim=1).clamp(min=1e-6)
    d_inv_sqrt = deg.pow(-0.5)
    return d_inv_sqrt[:, None] * a_hat * d_inv_sqrt[None, :]


# --------------------------------------------------------------------- #
# Asset Scoring Unit
# --------------------------------------------------------------------- #


class _GCNBlock(nn.Module):
    """One DeepTrader-style spatio-temporal block.

    Temporal conv (kernel 2 along T) followed by a GCN-style aggregation
    over assets, using the symmetric-normalised base adjacency plus a
    rank-1 learned adaptive adjacency (matching upstream's single
    learnable ``nodevec`` of shape (N, 1)).

    Args:
        in_channels: Input feature channels per (asset, time) entry.
        out_channels: Output feature channels per (asset, time) entry.
        num_assets: Number of assets in the universe (the N dimension).
        dropout: Dropout rate applied after each conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_assets: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.temporal = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 2),
            padding=(0, 1),
        )
        self.spatial = nn.Linear(out_channels, out_channels, bias=True)
        # Upstream form: a single learnable vector of shape (N, 1)
        # producing a near-rank-1 adaptive adjacency.
        self.nodevec = nn.Parameter(torch.randn(num_assets, 1) * 0.1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=False)

    def adaptive_adjacency(self) -> torch.Tensor:
        """Compute the learned ``softmax(relu(nodevec @ nodevec.T))`` adjacency.

        Returns:
            Tensor of shape ``(N, N)``.
        """
        logits = torch.relu(self.nodevec @ self.nodevec.T)
        return F.softmax(logits, dim=1)

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape ``(B, N, T, C_in)``.
            a_norm: Normalised base adjacency, shape ``(N, N)``.

        Returns:
            Tensor of shape ``(B, N, T, C_out)``.
        """
        # Temporal conv expects (B, C, N, T); move channels.
        h = x.permute(0, 3, 1, 2)
        h = self.temporal(h)
        # Trim the extra time step introduced by padding=(0, 1) so the
        # temporal dimension stays fixed at T.
        if h.shape[-1] != x.shape[2]:
            h = h[..., : x.shape[2]]
        h = self.activation(h)
        h = self.bn(h)
        h = h.permute(0, 2, 3, 1)  # (B, N, T, C_out)

        # Spatial aggregation: combine base and adaptive adjacency.
        a_total = 0.5 * (a_norm + self.adaptive_adjacency())
        # Aggregate along N: (B, N, T, C) -> (B, N, T, C).
        h_agg = torch.einsum("ij,bjtc->bitc", a_total, h)
        h_agg = self.spatial(h_agg)
        h_agg = self.activation(h_agg)
        return self.dropout(h_agg)


class ASU(nn.Module):
    """4-block adaptive GCN producing one sigmoid winner score per asset.

    Upstream form: the per-asset logit is passed through a sigmoid,
    giving scores in ``(0, 1)``. The short-book ranking criterion is
    ``sign(score) * (1 - score)`` (which for ``score in (0, 1)``
    simplifies to ``1 - score``).

    Args:
        num_assets: N, number of assets in the universe.
        in_features: Per-stock channels. Defaults to 6 (DeepTrader spec).
        hidden: Hidden channel size.
        num_blocks: Number of GCN blocks. Defaults to 4.
        dropout: Dropout rate inside each block.
    """

    def __init__(
        self,
        num_assets: int,
        in_features: int = 6,
        hidden: int = 32,
        num_blocks: int = 4,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_assets = num_assets
        self.input_proj = nn.Linear(in_features, hidden)
        self.blocks = nn.ModuleList(
            [
                _GCNBlock(
                    in_channels=hidden,
                    out_channels=hidden,
                    num_assets=num_assets,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Per-stock window, shape ``(B, N, T, F_asu)``.
            a_norm: Symmetric-normalised adjacency, shape ``(N, N)``.

        Returns:
            Per-asset sigmoid score tensor of shape ``(B, N)`` in ``(0, 1)``.
        """
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, a_norm)
        # Pool over time then map to a scalar per asset.
        h = h.mean(dim=2)  # (B, N, hidden)
        logits = self.head(h).squeeze(-1)  # (B, N)
        return torch.sigmoid(logits)


# --------------------------------------------------------------------- #
# Market Scoring Unit
# --------------------------------------------------------------------- #


class MSU(nn.Module):
    """LSTM + Bahdanau attention pooling over market features.

    Mirrors upstream ``src/model/MSU.py``:
    ``nn.LSTM(in, hidden) -> attn over L -> bn -> Linear(hidden, 2)``.
    Emits ``(mu, raw_sigma)`` per batch element. The caller is
    responsible for converting ``raw_sigma`` to ``sigma`` via softplus
    and sampling ``rho``.

    Args:
        in_features: Per-day market features. Defaults to 4.
        hidden: LSTM hidden size (== upstream ``hidden_dim``).
    """

    def __init__(
        self,
        in_features: int = 4,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden = hidden
        self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden)
        self.attn1 = nn.Linear(2 * hidden, hidden)
        self.attn2 = nn.Linear(hidden, 1)
        self.linear1 = nn.Linear(hidden, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.linear2 = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Market-feature window, shape ``(B, L, F_msu)``.

        Returns:
            (mu, raw_sigma), each shape ``(B,)``. The caller applies
            softplus to ``raw_sigma`` before sampling.
        """
        # LSTM expects (L, B, F); upstream permute(1, 0, 2).
        b, l, _ = x.shape
        x_t = x.permute(1, 0, 2)
        outputs, (h_n, _c_n) = self.lstm(x_t)  # outputs: (L, B, H)
        # Bahdanau attention: concat each step's output with the final
        # hidden state, project, tanh, project to scalar, softmax over L.
        h_rep = h_n.repeat(l, 1, 1)  # (L, B, H)
        scores = self.attn2(
            torch.tanh(self.attn1(torch.cat([outputs, h_rep], dim=2)))
        )  # (L, B, 1)
        scores = scores.squeeze(2).transpose(1, 0)  # (B, L)
        attn_weights = torch.softmax(scores, dim=1)  # (B, L)
        outputs_b = outputs.permute(1, 0, 2)  # (B, L, H)
        attn_embed = torch.bmm(
            attn_weights.unsqueeze(1), outputs_b
        ).squeeze(1)  # (B, H)
        embed = torch.relu(self.bn1(self.linear1(attn_embed)))
        params = self.linear2(embed)  # (B, 2)
        mu = params[:, 0]
        raw_sigma = params[:, 1]
        return mu, raw_sigma


# --------------------------------------------------------------------- #
# Portfolio generator and actor
# --------------------------------------------------------------------- #


def _top_g_softmax_long(scores: torch.Tensor, g: int) -> torch.Tensor:
    """Long-book softmax over top-G entries of ``scores`` (descending).

    Args:
        scores: Tensor of shape ``(B, N)``.
        g: Number of long picks per row.

    Returns:
        Sparse weight tensor of shape ``(B, N)``; nonzero entries sum
        to 1 along the asset axis.
    """
    values, indices = torch.topk(scores, k=g, dim=1)
    weights = torch.softmax(values, dim=1)
    out = torch.zeros_like(scores)
    out.scatter_(1, indices, weights)
    return out


def _top_g_softmax_short(scores: torch.Tensor, g: int) -> torch.Tensor:
    """Short-book softmax over top-G entries of ``sign(s)*(1-s)``.

    Mirrors upstream ``agent.py:__generator``:
    ``loser_scores = scores.sign() * (1 - scores)`` then
    ``softmax(topk(loser_scores))``.

    Args:
        scores: Per-asset sigmoid scores in ``(0, 1)``, shape ``(B, N)``.
        g: Number of short picks per row.

    Returns:
        Sparse weight tensor of shape ``(B, N)``; nonzero entries sum
        to 1 along the asset axis.
    """
    loser_scores = torch.sign(scores) * (1.0 - scores)
    values, indices = torch.topk(loser_scores, k=g, dim=1)
    weights = torch.softmax(values, dim=1)
    out = torch.zeros_like(scores)
    out.scatter_(1, indices, weights)
    return out


class DeepTraderActor(nn.Module):
    """ASU + MSU + Portfolio Generator composition.

    Args:
        num_assets: N, number of assets in the universe.
        top_g: Number of long picks (and short picks) per step.
            Upstream default is 4 for DJIA-30.
        asu_kwargs: Optional kwargs for the ASU sub-module.
        msu_kwargs: Optional kwargs for the MSU sub-module.
    """

    def __init__(
        self,
        num_assets: int,
        top_g: int = 4,
        asu_kwargs: Optional[Dict] = None,
        msu_kwargs: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.num_assets = num_assets
        self.top_g = top_g
        self.asu = ASU(num_assets=num_assets, **(asu_kwargs or {}))
        self.msu = MSU(**(msu_kwargs or {}))

    def forward(
        self,
        stocks_window: torch.Tensor,
        market_window: torch.Tensor,
        a_norm: torch.Tensor,
        stochastic: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for one batch of windows.

        Args:
            stocks_window: ``(B, N, T, F_asu)``.
            market_window: ``(B, L, F_msu)``.
            a_norm: Normalised base adjacency, ``(N, N)``.
            stochastic: If True sample leverage from the Gaussian
                policy (training); if False use ``clamp(mu, 0, 1)``.

        Returns:
            Dict with keys:
                scores: ``(B, N)`` ASU sigmoid scores in (0, 1).
                scores_p: ``(B, N)`` softmax(scores, dim=-1), the full
                    policy distribution used by the ASU surrogate.
                long_weights: ``(B, N)``.
                short_weights: ``(B, N)``.
                weights_2n: ``(B, 2N)`` concatenation of long then short.
                rho: ``(B,)`` leverage in [0, 1].
                log_prob_rho: ``(B,)`` Gaussian log-prob of the pre-clamp
                    sample (zeros when ``stochastic=False``).
        """
        scores = self.asu(stocks_window, a_norm)  # in (0, 1)
        long_weights = _top_g_softmax_long(scores, g=self.top_g)
        short_weights = _top_g_softmax_short(scores, g=self.top_g)
        weights_2n = torch.cat([long_weights, short_weights], dim=1)
        # Full-policy softmax used by the upstream ASU surrogate.
        scores_p = F.softmax(scores, dim=-1)

        mu, raw_sigma = self.msu(market_window)
        sigma = F.softplus(raw_sigma) + 1e-6
        if stochastic:
            m = Normal(mu, sigma)
            sample_rho = m.sample()
            rho = sample_rho.clamp(0.0, 1.0)
            log_prob_rho = m.log_prob(sample_rho)
        else:
            rho = mu.clamp(0.0, 1.0)
            log_prob_rho = torch.zeros_like(mu)

        return {
            "scores": scores,
            "scores_p": scores_p,
            "long_weights": long_weights,
            "short_weights": short_weights,
            "weights_2n": weights_2n,
            "rho": rho,
            "log_prob_rho": log_prob_rho,
        }


# --------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------- #


@dataclass
class DeepTraderEnvConfig:
    """Configuration for :class:`DeepTraderEnv`.

    Attributes:
        window: Look-back length L for both stock and market inputs.
            Upstream uses ``window_len=13`` weekly bars (from
            ``5*(window_len+1)`` calendar days). Our daily-granularity
            substitute is L=60 by default.
        fee_bps: Per-side transaction cost in basis points.
            Upstream ``fee=0.001`` corresponds to ``fee_bps=10.0``.
        trade_len: Rebalance interval in trading days. Upstream uses
            ``trade_len=21`` (monthly).
        episode_length: Maximum monthly steps per episode. Upstream uses
            ``max_steps=12``. ``None`` means run to the end of the panel.
    """

    window: int = 60
    fee_bps: float = 10.0
    trade_len: int = 21
    episode_length: Optional[int] = 12


class DeepTraderEnv:
    """Monthly-rebalance env over a (returns, stocks, market) panel.

    The env steps every ``trade_len`` trading days (default 21 = monthly).
    Each step the agent consumes a window of stock and market features,
    emits long and short weights plus leverage, and receives the
    realised ``trade_len``-day geometric return minus turnover cost.
    Intra-month, weights drift with daily returns (geometric
    compounding), matching upstream ``portfolio_env.PortfolioSim``.

    Args:
        returns: ``(T_total, N)`` ndarray of next-day per-asset simple
            returns (i.e. ``exp(log_ret) - 1``).
        stock_features: ``(T_total, N, F_asu)`` ndarray of per-stock
            features.
        market_features: ``(T_total, F_msu)`` ndarray of market features.
        config: :class:`DeepTraderEnvConfig`.
    """

    def __init__(
        self,
        returns: np.ndarray,
        stock_features: np.ndarray,
        market_features: np.ndarray,
        config: Optional[DeepTraderEnvConfig] = None,
    ) -> None:
        self.config = config or DeepTraderEnvConfig()
        assert returns.ndim == 2
        assert stock_features.ndim == 3
        assert market_features.ndim == 2
        assert (
            returns.shape[0]
            == stock_features.shape[0]
            == market_features.shape[0]
        )
        self.returns = returns.astype(np.float32)
        self.stock_features = stock_features.astype(np.float32)
        self.market_features = market_features.astype(np.float32)
        self.t_total, self.num_assets = returns.shape
        self._t = self.config.window
        self._prev_weights = np.zeros(
            2 * self.num_assets, dtype=np.float32
        )
        self._step_count = 0

    @property
    def num_steps(self) -> int:
        """Maximum number of monthly steps before the panel runs out."""
        return max(
            0,
            (self.t_total - self.config.window - 1)
            // self.config.trade_len,
        )

    def reset(self, start_t: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Reset the env to the first valid step.

        Args:
            start_t: Optional starting day index. Defaults to the window
                size so the first observation has a complete history.

        Returns:
            Observation dict (see :meth:`_observation`).
        """
        self._t = (
            start_t if start_t is not None else self.config.window
        )
        self._prev_weights = np.zeros(
            2 * self.num_assets, dtype=np.float32
        )
        self._step_count = 0
        return self._observation()

    def _observation(self) -> Dict[str, np.ndarray]:
        """Build the current observation window."""
        t = self._t
        w = self.config.window
        return {
            "stocks_window": self.stock_features[t - w : t],
            "market_window": self.market_features[t - w : t],
        }

    def step(
        self,
        weights_2n: np.ndarray,
        rho: float,
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, float]]:
        """Apply one monthly rebalance and return ``(obs, reward, done, info)``.

        The agent's weights are held constant for ``trade_len`` trading
        days, drifting with daily returns. The realised return for the
        month is the long-book geometric return minus the short-book
        geometric return, scaled by ``rho``. Turnover cost is paid at the
        time of rebalance using ``weights_2n - prev_weights``.

        Args:
            weights_2n: Length-``2N`` weight vector (long then short).
            rho: Scalar leverage in [0, 1].

        Returns:
            Tuple ``(obs, reward, done, info)``. ``info`` carries the
            decomposed long/short geometric returns and turnover cost.
            ``info["future_ror"]`` is the ``(N,)`` per-asset
            ``trade_len``-day simple return used by the ASU surrogate.
        """
        assert weights_2n.shape == (2 * self.num_assets,)
        long_w = weights_2n[: self.num_assets]
        short_w = weights_2n[self.num_assets :]

        t = self._t
        tl = self.config.trade_len
        end = min(t + tl, self.t_total)
        # Per-asset simple-return series over the holding period.
        period_rets = self.returns[t:end]  # (<=tl, N)

        # Geometric compounding per asset over the holding period:
        # future_ror_n = prod(1 + r_{t..t+tl}) per asset.
        future_ror = np.prod(1.0 + period_rets, axis=0).astype(np.float32)

        long_geo = float((long_w * future_ror).sum())
        short_geo = float((short_w * future_ror).sum())
        # Long P&L = long_geo - 1; short P&L = 1 - short_geo (paying back
        # at a lower price is profit). Leveraged by rho.
        port_ret = float(rho) * (long_geo - short_geo)
        turnover = float(np.abs(weights_2n - self._prev_weights).sum())
        fee = self.config.fee_bps * 1e-4 * turnover
        reward = port_ret - fee

        self._prev_weights = weights_2n.astype(np.float32)
        self._t = end
        self._step_count += 1
        max_steps = (
            self.config.episode_length
            if self.config.episode_length is not None
            else self.num_steps
        )
        done = (
            self._t >= self.t_total - 1
            or self._step_count >= max_steps
            or self._t + tl > self.t_total
        )
        info = {
            "port_ret": port_ret,
            "long_geo": long_geo,
            "short_geo": short_geo,
            "turnover": turnover,
            "fee": fee,
            "future_ror": future_ror,
        }
        return self._observation(), reward, done, info


# --------------------------------------------------------------------- #
# REINFORCE training loop (batched, upstream-faithful)
# --------------------------------------------------------------------- #


@dataclass
class DeepTraderTrainConfig:
    """Training-loop configuration.

    Attributes:
        epochs: Number of REINFORCE epochs.
        lr: Adam learning rate. Upstream default ``1e-6``.
        weight_decay: Adam weight decay. Upstream default ``1e-3``.
        gamma: Weighting of the MSU gradient relative to the ASU
            gradient. Upstream default ``0.05``.
        batch_size: Parallel trajectories per epoch. Upstream default
            ``37``.
        rollout_steps: Number of monthly steps per trajectory. Upstream
            default ``max_steps=12`` (i.e. 12 months).
        grad_clip: Gradient norm clip. Upstream default ``100.0``.
        eval_every: Run a deterministic eval every N epochs.
    """

    epochs: int = 500
    lr: float = 1e-6
    weight_decay: float = 1e-3
    gamma: float = 0.05
    batch_size: int = 37
    rollout_steps: int = 12
    grad_clip: Optional[float] = 100.0
    eval_every: int = 50


def _max_drawdown_curve(equity: torch.Tensor) -> torch.Tensor:
    """Per-row max drawdown of a 2-D equity-curve tensor.

    Args:
        equity: Tensor of shape ``(B, L+1)`` with cumulative equity
            values per trajectory.

    Returns:
        Tensor of shape ``(B,)`` of non-negative MDD values in [0, 1].
    """
    running_max = torch.cummax(equity, dim=1).values
    drawdown = (running_max - equity) / running_max.clamp(min=1e-6)
    return drawdown.max(dim=1).values


def _build_starts(env: DeepTraderEnv, batch_size: int) -> np.ndarray:
    """Sample ``batch_size`` valid starting indices for parallel rollouts.

    Args:
        env: The training environment.
        batch_size: Number of trajectories.

    Returns:
        ``(batch_size,)`` int64 array of starting day indices.
    """
    w = env.config.window
    tl = env.config.trade_len
    max_steps = (
        env.config.episode_length
        if env.config.episode_length is not None
        else env.num_steps
    )
    last_valid = env.t_total - tl * max_steps - 1
    last_valid = max(w + 1, last_valid)
    return np.random.randint(
        w, last_valid + 1, size=batch_size,
    ).astype(np.int64)


def _rollout_batch(
    actor: DeepTraderActor,
    env: DeepTraderEnv,
    a_norm: torch.Tensor,
    rollout_steps: int,
    device: torch.device,
    starts: np.ndarray,
) -> Dict[str, torch.Tensor]:
    """Roll out ``batch_size`` parallel trajectories on the env.

    Each trajectory starts at a different day index. At each monthly
    step we form a batch of ``(stocks_window, market_window)`` tensors
    of shape ``(B, ...)``, call the actor once, and apply each row's
    weights to its own env-state copy. We aggregate the per-step
    quantities into the dict returned.

    Args:
        actor: DeepTrader actor.
        env: A template env used for its shape and config; we do NOT
            mutate its cursor inside this function.
        a_norm: Normalised adjacency, ``(N, N)`` on ``device``.
        rollout_steps: Monthly steps per trajectory.
        device: Torch device for tensor ops.
        starts: ``(batch_size,)`` int64 starting indices.

    Returns:
        Dict with keys:
            asu_step_losses: ``(B, S)`` ``log(sum_n p_n * z_n)``.
            msu_log_probs: ``(B, S)`` Gaussian log-prob of rho.
            rewards: ``(B, S)`` per-step net returns.
            mkt_avg_returns: ``(B, S)`` cross-sectional mean of
                next-period returns (the upstream "market_avg_return").
            equity: ``(B, S+1)`` cumulative equity per trajectory.
            done_mask: ``(B, S)`` bool, 1 where the step was real
                (not past end-of-panel).
    """
    batch_size = len(starts)
    cfg = env.config
    w = cfg.window
    tl = cfg.trade_len

    # Each trajectory has its own cursor and prev_weights.
    t = starts.copy()
    prev_w = np.zeros(
        (batch_size, 2 * env.num_assets), dtype=np.float32,
    )

    asu_losses: List[torch.Tensor] = []
    msu_lps: List[torch.Tensor] = []
    rewards_per_step: List[torch.Tensor] = []
    mkt_avg_per_step: List[torch.Tensor] = []
    equity_vals = torch.ones(batch_size, device=device)
    equity_list: List[torch.Tensor] = [equity_vals]
    alive_list: List[torch.Tensor] = []

    for _step in range(rollout_steps):
        # Build batched windows. For rows that ran out of data, we clip
        # the cursor to t_total - 1 and mask them with alive=False.
        alive_np = np.array(
            [bool((t[i] + tl <= env.t_total) and (t[i] >= w))
             for i in range(batch_size)],
            dtype=np.bool_,
        )
        alive = torch.from_numpy(alive_np).to(device)
        if not alive.any():
            break
        t_safe = np.clip(t, w, env.t_total - 1)
        stocks = np.stack(
            [
                env.stock_features[ti - w : ti].transpose(1, 0, 2)
                for ti in t_safe
            ],
            axis=0,
        )  # (B, N, T, F_asu)
        market = np.stack(
            [env.market_features[ti - w : ti] for ti in t_safe], axis=0,
        )  # (B, L, F_msu)
        stocks_t = torch.from_numpy(stocks).to(device)
        market_t = torch.from_numpy(market).to(device)

        out = actor(stocks_t, market_t, a_norm, stochastic=True)
        scores_p = out["scores_p"]  # (B, N)
        long_w = out["long_weights"]
        short_w = out["short_weights"]
        weights_2n = out["weights_2n"]
        rho = out["rho"]
        log_prob_rho = out["log_prob_rho"]

        weights_np = weights_2n.detach().cpu().numpy()
        rho_np = rho.detach().cpu().numpy()

        # Compute per-asset next-period returns per row (geometric over
        # trade_len). This is the upstream ``ror`` variable.
        future_ror_rows = np.zeros(
            (batch_size, env.num_assets), dtype=np.float32,
        )
        port_ret = np.zeros(batch_size, dtype=np.float32)
        fee_arr = np.zeros(batch_size, dtype=np.float32)
        mkt_avg = np.zeros(batch_size, dtype=np.float32)
        new_alive = alive.detach().cpu().numpy().copy()
        for i in range(batch_size):
            if not new_alive[i]:
                continue
            ti = int(t[i])
            end = min(ti + tl, env.t_total)
            period = env.returns[ti:end]
            fr = np.prod(1.0 + period, axis=0)
            future_ror_rows[i] = fr
            lw = weights_np[i, : env.num_assets]
            sw = weights_np[i, env.num_assets :]
            long_geo = float((lw * fr).sum())
            short_geo = float((sw * fr).sum())
            pr = float(rho_np[i]) * (long_geo - short_geo)
            turn = float(np.abs(weights_np[i] - prev_w[i]).sum())
            fee = cfg.fee_bps * 1e-4 * turn
            port_ret[i] = pr - fee
            fee_arr[i] = fee
            mkt_avg[i] = float(fr.mean()) - 1.0
            prev_w[i] = weights_np[i]

        # Cross-sectional z-score of next-period returns at this step.
        ror_t = torch.from_numpy(future_ror_rows).to(device)
        ror_mean = ror_t.mean(dim=-1, keepdim=True)
        ror_std = ror_t.std(dim=-1, keepdim=True).clamp(min=1e-6)
        z_ror = (ror_t - ror_mean) / ror_std  # (B, N)

        # Upstream ASU surrogate: log(sum_n softmax(scores)_n * z_n).
        inner = (z_ror * scores_p).sum(dim=-1)  # (B,)
        # log requires a positive argument; the upstream code allows
        # NaN/inf to surface (`assert not torch.isnan(loss)`). We clamp
        # to a tiny positive to keep the loss finite for runs where the
        # policy briefly puts all mass on below-mean stocks.
        asu_step_loss = torch.log(inner.clamp(min=1e-8))
        asu_losses.append(asu_step_loss)
        msu_lps.append(log_prob_rho)
        rewards_per_step.append(
            torch.from_numpy(port_ret).to(device)
        )
        mkt_avg_per_step.append(torch.from_numpy(mkt_avg).to(device))
        alive_list.append(alive)

        # Update equity (multiplicative for live rows, identity for
        # dead rows).
        port_ret_t = torch.from_numpy(port_ret).to(device)
        new_eq = equity_vals * (1.0 + port_ret_t * alive.float())
        equity_vals = new_eq
        equity_list.append(equity_vals)

        # Advance cursors.
        t = t + tl

    if not asu_losses:
        # No live steps at all; return empties so caller can no-op.
        empty = torch.zeros((batch_size, 0), device=device)
        return {
            "asu_step_losses": empty,
            "msu_log_probs": empty,
            "rewards": empty,
            "mkt_avg_returns": empty,
            "equity": equity_vals.unsqueeze(1),
            "done_mask": torch.zeros(
                (batch_size, 0), dtype=torch.bool, device=device,
            ),
        }

    return {
        "asu_step_losses": torch.stack(asu_losses, dim=1),  # (B, S)
        "msu_log_probs": torch.stack(msu_lps, dim=1),  # (B, S)
        "rewards": torch.stack(rewards_per_step, dim=1),  # (B, S)
        "mkt_avg_returns": torch.stack(mkt_avg_per_step, dim=1),  # (B,S)
        "equity": torch.stack(equity_list, dim=1),  # (B, S+1)
        "done_mask": torch.stack(alive_list, dim=1),  # (B, S)
    }


def train_deeptrader(
    actor: DeepTraderActor,
    env: DeepTraderEnv,
    *,
    adjacency: np.ndarray,
    device: Optional[torch.device] = None,
    val_env: Optional[DeepTraderEnv] = None,
    config: Optional[DeepTraderTrainConfig] = None,
    verbose: bool = False,
) -> Tuple[Dict[str, torch.Tensor], List[float]]:
    """Batched REINFORCE training loop for DeepTrader.

    Each epoch runs ``batch_size`` parallel trajectories starting at
    different days, then averages the upstream-form surrogate over the
    batch.

    Args:
        actor: :class:`DeepTraderActor`.
        env: Training :class:`DeepTraderEnv`.
        adjacency: ``(N, N)`` numpy adjacency for the GCN.
        device: Torch device.
        val_env: Optional separate val env for selecting the best state.
        config: :class:`DeepTraderTrainConfig`.
        verbose: If True print epoch-level summaries.

    Returns:
        ``(best_state_dict, loss_history)``.
    """
    cfg = config or DeepTraderTrainConfig()
    device = device or torch.device("cpu")
    actor.to(device)
    adj_t = torch.from_numpy(adjacency.astype(np.float32)).to(device)
    a_norm = symmetric_normalize(adj_t)

    optim = torch.optim.Adam(
        actor.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    loss_history: List[float] = []
    best_val = -float("inf")
    best_state = {
        k: v.detach().clone() for k, v in actor.state_dict().items()
    }

    for epoch in range(cfg.epochs):
        actor.train()
        optim.zero_grad()
        starts = _build_starts(env, cfg.batch_size)
        traj = _rollout_batch(
            actor, env, a_norm, cfg.rollout_steps, device, starts,
        )
        asu = traj["asu_step_losses"]  # (B, S)
        msu_lp = traj["msu_log_probs"]  # (B, S)
        rewards = traj["rewards"]  # (B, S)
        mkt_avg = traj["mkt_avg_returns"]  # (B, S)
        equity = traj["equity"]  # (B, S+1)
        alive = traj["done_mask"].float()  # (B, S)

        if asu.numel() == 0:
            if verbose:
                print(
                    f"[deeptrader] epoch {epoch + 1}/{cfg.epochs} "
                    f"NO_LIVE_STEPS",
                    flush=True,
                )
            loss_history.append(float("nan"))
            continue

        # Upstream batch normalisation of rewards (subtract market avg
        # then z-score across the batch dim per step).
        adv = rewards - mkt_avg
        adv_mean = adv.mean(dim=0, keepdim=True)
        adv_std = adv.std(dim=0, keepdim=True).clamp(min=1e-6)
        adv_norm = (adv - adv_mean) / adv_std  # (B, S); used for diag

        # MSU reward: -2 * (mdd - 0.5), broadcast over steps.
        mdd = _max_drawdown_curve(equity)  # (B,)
        rewards_mdd = -2.0 * (mdd - 0.5)  # (B,)
        # Broadcast to (B, S) via outer product with alive mask.
        rewards_mdd_step = rewards_mdd.unsqueeze(1) * alive  # (B, S)

        gradient_asu = asu * alive  # (B, S)
        gradient_rho = rewards_mdd_step * msu_lp  # (B, S)
        # Upstream form: loss = -(gamma * gradient_rho + gradient_asu).
        loss_per_step = -(cfg.gamma * gradient_rho + gradient_asu)
        # Average over alive steps.
        n_alive = alive.sum().clamp(min=1.0)
        loss = loss_per_step.sum() / n_alive

        # Use adv_norm to log a diagnostic ASU-advantage scalar without
        # affecting the optimiser (matching the spirit of the upstream
        # rewards_total normalisation that they use elsewhere).
        _ = adv_norm.detach()

        assert not torch.isnan(loss), "deeptrader loss is NaN"
        loss.backward()
        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                actor.parameters(), cfg.grad_clip,
            )
        optim.step()
        loss_history.append(float(loss.detach().cpu()))

        if val_env is not None and (epoch + 1) % cfg.eval_every == 0:
            val_metrics = evaluate_deeptrader(
                actor, val_env, adjacency, device,
            )
            score = val_metrics["sharpe"]
            if math.isfinite(score) and score > best_val:
                best_val = score
                best_state = {
                    k: v.detach().clone()
                    for k, v in actor.state_dict().items()
                }
            if verbose:
                print(
                    f"[deeptrader] epoch {epoch + 1}/{cfg.epochs} "
                    f"loss={loss_history[-1]:.6f} "
                    f"val_sharpe={score:.4f}",
                    flush=True,
                )
        elif verbose and (epoch + 1) % max(1, cfg.epochs // 10) == 0:
            print(
                f"[deeptrader] epoch {epoch + 1}/{cfg.epochs} "
                f"loss={loss_history[-1]:.6f}",
                flush=True,
            )

    if val_env is None:
        best_state = {
            k: v.detach().clone() for k, v in actor.state_dict().items()
        }
    return best_state, loss_history


# --------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------- #


def evaluate_deeptrader(
    actor: DeepTraderActor,
    env: DeepTraderEnv,
    adjacency: np.ndarray,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Deterministic rollout over the env; return daily-Sharpe metrics.

    The policy rebalances every ``trade_len`` trading days (monthly by
    default). For Sharpe stability, we report DAILY-granularity returns:
    intra-month, the portfolio weights drift with daily asset returns
    (geometric compounding) and we record the daily portfolio return
    minus its share of the rebalance-day turnover fee. Sharpe and
    ann_return are annualised at the daily cadence ``sqrt(252)``.

    Rationale: short test segments (e.g. F5 with ~123 days) produce only
    2-3 monthly rebalance periods, which gives Sharpe a microscopic
    denominator and numerically explodes the ratio. Reporting daily
    returns under the same monthly-rebalance policy is the apples-to-
    apples comparison against the other baselines (which all eval daily)
    and is numerically stable.

    Args:
        actor: Trained :class:`DeepTraderActor`.
        env: :class:`DeepTraderEnv` to evaluate on.
        adjacency: ``(N, N)`` numpy adjacency.
        device: Torch device.

    Returns:
        Dict with keys ``sharpe``, ``ann_return``, ``ann_vol``,
        ``final_equity``, ``n_steps`` (number of daily returns).
    """
    device = device or torch.device("cpu")
    actor.to(device)
    actor.eval()
    adj_t = torch.from_numpy(adjacency.astype(np.float32)).to(device)
    a_norm = symmetric_normalize(adj_t)
    cfg = env.config
    tl = cfg.trade_len
    w = cfg.window
    n_assets = env.num_assets
    fee_rate = cfg.fee_bps * 1e-4

    daily_returns: List[float] = []
    prev_w = np.zeros(2 * n_assets, dtype=np.float32)
    t = w
    cur_long: Optional[np.ndarray] = None
    cur_short: Optional[np.ndarray] = None
    cur_rho: float = 0.0

    with torch.no_grad():
        while t + 1 <= env.t_total:
            # On rebalance days the policy emits new (long, short, rho).
            days_since_rebal = (t - w) % tl
            if days_since_rebal == 0:
                stocks = (
                    torch.from_numpy(env.stock_features[t - w : t])
                    .unsqueeze(0)
                    .to(device)
                )
                # (1, T, N, F) -> (1, N, T, F).
                stocks = stocks.permute(0, 2, 1, 3)
                market = (
                    torch.from_numpy(env.market_features[t - w : t])
                    .unsqueeze(0)
                    .to(device)
                )
                out = actor(
                    stocks, market, a_norm, stochastic=False,
                )
                weights_2n = out["weights_2n"].squeeze(0).cpu().numpy()
                cur_long = weights_2n[:n_assets].copy()
                cur_short = weights_2n[n_assets:].copy()
                cur_rho = float(out["rho"].squeeze(0).cpu())
                # Fee paid on this rebalance day (one-off).
                turnover = float(np.abs(weights_2n - prev_w).sum())
                prev_w = weights_2n.astype(np.float32)
                fee_today = fee_rate * turnover
            else:
                fee_today = 0.0

            # Daily portfolio return: long_w . r - short_w . r, scaled
            # by rho. cur_long/cur_short have drifted multiplicatively
            # with the daily returns since the last rebalance, mirroring
            # the geometric-compounding contract.
            r_t = env.returns[t]
            long_ret = float((cur_long * r_t).sum()) if cur_long is not None else 0.0
            short_ret = float((cur_short * r_t).sum()) if cur_short is not None else 0.0
            port_ret = cur_rho * (long_ret - short_ret) - fee_today
            daily_returns.append(port_ret)

            # Drift weights for next day.
            if cur_long is not None:
                growth_l = 1.0 + r_t
                cur_long = (cur_long * growth_l)
                s = cur_long.sum()
                if abs(s) > 1e-12:
                    cur_long = cur_long / s
                cur_short = (cur_short * growth_l)
                s2 = cur_short.sum()
                if abs(s2) > 1e-12:
                    cur_short = cur_short / s2

            t += 1

    actor.train()
    arr = np.array(daily_returns, dtype=np.float64)
    if arr.size == 0:
        return {
            "sharpe": float("nan"),
            "ann_return": float("nan"),
            "ann_vol": float("nan"),
            "final_equity": 1.0,
            "n_steps": 0,
        }
    mean = arr.mean()
    std = arr.std(ddof=1) if arr.size > 1 else 0.0
    sharpe = (
        (mean / std) * math.sqrt(252.0)
        if std > 1e-12 else 0.0
    )
    ann_return = mean * 252.0
    ann_vol = std * math.sqrt(252.0)
    final_equity = float(np.prod(1.0 + arr))
    return {
        "sharpe": float(sharpe),
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "final_equity": final_equity,
        "n_steps": int(arr.size),
    }


# --------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------- #


def _build_synthetic_panel(
    num_assets: int = 5,
    t_total: int = 600,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Synthetic toy panel used by :func:`smoke_test`.

    Args:
        num_assets: N.
        t_total: Total time steps.
        seed: RNG seed.

    Returns:
        Dict with keys ``returns``, ``stock_features``,
        ``market_features``, ``adjacency``.
    """
    rng = np.random.default_rng(seed)
    returns = 0.001 + 0.01 * rng.standard_normal((t_total, num_assets))
    stock_features = rng.standard_normal(
        (t_total, num_assets, 6),
    ).astype(np.float32)
    market_features = rng.standard_normal((t_total, 4)).astype(np.float32)
    # Two-sector toy adjacency.
    half = num_assets // 2
    adj = np.zeros((num_assets, num_assets), dtype=np.float32)
    adj[:half, :half] = 1.0
    adj[half:, half:] = 1.0
    np.fill_diagonal(adj, 1.0)
    return {
        "returns": returns.astype(np.float32),
        "stock_features": stock_features,
        "market_features": market_features,
        "adjacency": adj,
    }


def smoke_test() -> Dict[str, float]:
    """10-epoch CPU smoke on 5-asset synthetic data.

    Verifies the model trains end-to-end without NaN/inf and that the
    eval helper returns a finite Sharpe under the new monthly-rebalance
    contract.

    Returns:
        Dict with the final eval metrics plus the recorded loss history.
    """
    torch.manual_seed(0)
    np.random.seed(0)

    panel = _build_synthetic_panel(num_assets=5, t_total=600)
    env = DeepTraderEnv(
        returns=panel["returns"],
        stock_features=panel["stock_features"],
        market_features=panel["market_features"],
        config=DeepTraderEnvConfig(
            window=20, fee_bps=10.0, trade_len=21, episode_length=6,
        ),
    )
    val_env = DeepTraderEnv(
        returns=panel["returns"][300:],
        stock_features=panel["stock_features"][300:],
        market_features=panel["market_features"][300:],
        config=DeepTraderEnvConfig(
            window=20, fee_bps=10.0, trade_len=21, episode_length=6,
        ),
    )
    actor = DeepTraderActor(
        num_assets=5,
        top_g=2,
        asu_kwargs={"hidden": 8, "num_blocks": 2},
        msu_kwargs={"hidden": 16},
    )
    cfg = DeepTraderTrainConfig(
        epochs=10,
        lr=1e-3,
        batch_size=4,
        rollout_steps=6,
        eval_every=5,
        grad_clip=100.0,
    )
    best_state, losses = train_deeptrader(
        actor,
        env,
        adjacency=panel["adjacency"],
        val_env=val_env,
        config=cfg,
        verbose=True,
    )
    actor.load_state_dict(best_state)
    metrics = evaluate_deeptrader(actor, val_env, panel["adjacency"])
    print(f"[smoke] losses: {losses}", flush=True)
    print(f"[smoke] eval: {metrics}", flush=True)
    assert all(math.isfinite(x) for x in losses), (
        "Non-finite loss in smoke"
    )
    assert math.isfinite(metrics["sharpe"]), (
        "Non-finite sharpe in smoke"
    )
    return {**metrics, "loss_history": losses}


if __name__ == "__main__":
    smoke_test()
