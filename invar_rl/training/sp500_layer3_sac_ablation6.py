"""S&P 500 Phase 7 Layer 3 SAC Ablation 6: macro-conditional concentration.

Tests whether a 2-D SAC action (exposure scalar + concentration logit
mapped to a daily top-K) beats the canonical 1-D exposure scalar with
fixed K=50. Mirrors :mod:`invar_rl.training.stage3_rl_canonical` for
the canonical SP500 Option A pipeline (Layer 1 canonical InVAR + fixed
equal-weight top-K L/S wrapper + Layer 2 SAC; the canonical wrapper is
top-50 per side) with exactly one change: the SAC action becomes
``(e_t, k_logit_t)`` in ``[0, 1.5] x [-5, 5]``, where

    K_t = round(K_MIN + (K_MAX - K_MIN) * sigmoid(k_logit_t))

with ``K_MIN = 5`` and ``K_MAX = 50``. The wrapper picks the top-K_t
long and bottom-K_t short by InVAR score each day, equal-weight,
dollar-neutral, daily rebalance, before applying the exposure scalar.

This is still within Option A's information-transfer framing: the
agent has no portfolio-optimisation freedom (weights remain equal),
only a macro-conditional concentration choice on top of exposure.

For each ``(fold, seed)`` cell:

1. Build the SP500 ``lattice_native`` bridge for the fold.
2. Load the canonical InVAR full state_dict from
   ``invar_rl/results/stage1/_ckpt/foldF_seedS_full.pt``.
3. Precompute, per day, the top-50 long and bottom-50 short forward
   returns sorted by InVAR score plus the Layer 2 / risk observation
   fields. The tape carries enough to compute strategy return for any
   K in [5, 50].
4. Train SAC for ``--total-timesteps`` env steps with the 2-D action;
   per ``--eval-freq`` steps roll out on the 2017-H2 + 2018-H2 val
   segment and keep the best-val-Sharpe checkpoint.
5. Restore the best-val ckpt, roll out deterministically on the test
   segment, and persist per-day exposure / K_t / strategy return parquet
   plus a per-cell summary JSON with pooled Sharpe AND mean K_t per
   fold so the macro-conditional concentration behaviour can be
   analysed downstream.

CLI::

    python -m invar_rl.training.sp500_layer3_sac_ablation6 \\
        --fold F --seed S \\
        --total-timesteps 20000 \\
        --output-dir-root outputs/sp500/ablations/ablation6
"""
from __future__ import annotations

import argparse
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.invar import InVARConfig

from invar_rl.common.config import load_layer3_config, load_stage3_config
from invar_rl.common.seeding import set_global_seed
from invar_rl.data.lattice_bridge import build_lattice_bridge
from invar_rl.layer1_ranker.canonical_runner import load_trained_invar
from invar_rl.layer3_control.observation import RiskState, observation_dim
from invar_rl.layer3_control.precompute import EpisodeTape


# Canonical Option A wrapper concentration: top-50 per side.
_K_MAX: int = 50
_K_MIN: int = 5
_TRADING_DAYS: int = 252

# Volatility-window + regime-change Z used by the canonical env;
# kept private here for behaviour parity with ExposureEnv.
_VOL_WINDOW = 20
_REGIME_Z = 3.0


@dataclass
class VariableKTape:
    """Per-day data needed to compute strategy return for any K in [K_MIN, K_MAX].

    The ``base_return`` and ``base_gross`` fields of the parent EpisodeTape
    are still populated (with the fixed top-K_MAX wrapper return) so that
    the canonical observation pipeline and risk-state plumbing reuse
    works byte-for-byte; only the env step overrides the realised return
    using the action-chosen K_t.

    Attributes:
        episode: A standard EpisodeTape whose ``base_return`` /
            ``base_gross`` correspond to the fixed K=K_MAX wrapper. Reused
            unchanged for observation construction.
        long_returns: Per-day per-rank forward log-return of the top-K_MAX
            long names (rank 0 = highest score), shape ``(T, K_MAX)``.
        short_returns: Per-day per-rank forward log-return of the bottom-K_MAX
            short names (rank 0 = lowest score), shape ``(T, K_MAX)``.
        long_counts: Per-day count of valid long names available (may be
            below K_MAX in thin cross-sections), shape ``(T,)`` int.
        short_counts: Per-day count of valid short names available, shape
            ``(T,)`` int.
    """

    episode: EpisodeTape
    long_returns: np.ndarray
    short_returns: np.ndarray
    long_counts: np.ndarray
    short_counts: np.ndarray

    def __len__(self) -> int:
        return len(self.episode)

    @property
    def macro_dim(self) -> int:
        return self.episode.macro_dim


def precompute_variable_k_tape(
    bundle,
    bridge,
    day_indices,
    stride: int = 1,
    k_max: int = _K_MAX,
) -> VariableKTape:
    """Build a VariableKTape from canonical InVAR scores + bridge returns.

    Mirrors the structure of
    :func:`invar_rl.layer3_control.precompute_canonical.precompute_tape_canonical`
    but persists per-day per-rank forward returns for the top-``k_max``
    long + bottom-``k_max`` short names so the env can compute strategy
    return for any K in ``[K_MIN, k_max]``.

    The canonical fixed K=``k_max`` wrapper return is also stored in the
    embedded ``EpisodeTape.base_return`` so the observation / risk-state
    pipeline reuses :func:`invar_rl.layer3_control.observation.build_observation`
    unchanged.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    day_indices = list(day_indices)[::stride]

    days: List[int] = []
    disp: List[float] = []
    enc: List[np.ndarray] = []
    pvol: List[float] = []
    effn: List[float] = []
    bret: List[float] = []
    bgross: List[float] = []
    ic: List[float] = []
    long_rets: List[np.ndarray] = []
    short_rets: List[np.ndarray] = []
    long_n: List[int] = []
    short_n: List[int] = []

    for day in day_indices:
        if day < bridge.temporal_window - 1:
            continue
        if day + 1 >= bridge.log_returns_1d.shape[0]:
            continue
        active_global = np.nonzero(bridge.tradable[day])[0]
        if active_global.size < 2 * _K_MIN:
            continue
        out = bundle.forward_day(day)
        scores = out["scores"].detach().cpu().numpy().astype(np.float64)
        macro_enc_day = out["macro_input"].cpu().numpy().astype(np.float64)
        score_dispersion_day = float(out["score_dispersion"].cpu().item())
        if scores.size != active_global.size:
            active_global = out["active_indices"].cpu().numpy().astype(
                np.int64
            )
        next_ret = bridge.log_returns_1d[day + 1, active_global]
        next_ret = np.where(np.isfinite(next_ret), next_ret, 0.0)

        # Rank names by score ascending; longs = highest, shorts = lowest.
        order_asc = np.argsort(scores)
        n_active = scores.size
        n_long = min(k_max, n_active)
        n_short = min(k_max, n_active - n_long)  # disjoint long/short sets
        long_idx = order_asc[-n_long:][::-1]      # rank 0 = highest score
        short_idx = order_asc[:n_short]           # rank 0 = lowest score

        long_arr = np.zeros(k_max, dtype=np.float64)
        short_arr = np.zeros(k_max, dtype=np.float64)
        long_arr[:n_long] = next_ret[long_idx]
        short_arr[:n_short] = next_ret[short_idx]

        # Canonical fixed K=k_max wrapper return for the embedded EpisodeTape.
        k_fixed = min(k_max, n_long, n_short)
        if k_fixed == 0:
            continue
        port_ret = float(
            long_arr[:k_fixed].mean() - short_arr[:k_fixed].mean()
        )

        days.append(int(day))
        disp.append(score_dispersion_day)
        enc.append(macro_enc_day)
        pvol.append(0.0)  # no QP -> no predicted vol; consistent with equal_topk path
        # Equal-weight L/S with 2K names, |w|=1/(2K) -> eff positions = 2K.
        effn.append(float(2 * k_fixed))
        bret.append(port_ret)
        bgross.append(1.0)
        long_rets.append(long_arr)
        short_rets.append(short_arr)
        long_n.append(int(n_long))
        short_n.append(int(n_short))

        if scores.size >= 2 and np.std(scores) > 0 and np.std(next_ret) > 0:
            rs = np.argsort(np.argsort(scores)).astype(np.float64)
            rr = np.argsort(np.argsort(next_ret)).astype(np.float64)
            ic.append(float(np.corrcoef(rs, rr)[0, 1]))
        else:
            ic.append(0.0)

    episode = EpisodeTape(
        days=np.asarray(days, dtype=np.int64),
        score_dispersion=np.asarray(disp, dtype=np.float64),
        macro_encoding=np.asarray(enc, dtype=np.float64),
        pred_vol=np.asarray(pvol, dtype=np.float64),
        eff_positions=np.asarray(effn, dtype=np.float64),
        base_return=np.asarray(bret, dtype=np.float64),
        base_gross=np.asarray(bgross, dtype=np.float64),
        daily_ic=np.asarray(ic, dtype=np.float64),
    )
    return VariableKTape(
        episode=episode,
        long_returns=np.asarray(long_rets, dtype=np.float64),
        short_returns=np.asarray(short_rets, dtype=np.float64),
        long_counts=np.asarray(long_n, dtype=np.int64),
        short_counts=np.asarray(short_n, dtype=np.int64),
    )


def _k_from_logit(logit: float) -> int:
    """Map a logit in [-5, 5] to an integer K in [K_MIN, K_MAX] via sigmoid."""
    s = 1.0 / (1.0 + float(np.exp(-float(logit))))
    k = int(round(_K_MIN + (_K_MAX - _K_MIN) * s))
    return int(np.clip(k, _K_MIN, _K_MAX))


import gymnasium as gym
from gymnasium import spaces


class VariableKExposureEnv(gym.Env):
    """Gymnasium env with 2-D action: exposure scalar + concentration logit.

    Wraps a :class:`VariableKTape`. Observation pipeline (and therefore
    the SB3 SAC compatibility) is identical to
    :class:`invar_rl.layer3_control.env.ExposureEnv`; only the action
    space and the per-step strategy return computation differ.

    Action layout: ``(exposure, k_logit)`` where ``exposure`` is in
    ``[exposure_min, exposure_max]`` (canonical 0.0 / 1.5) and ``k_logit``
    is in ``[-5, 5]``. ``K_t = round(K_MIN + (K_MAX - K_MIN) * sigmoid(k_logit))``,
    clipped to ``[K_MIN, K_MAX]`` (``5..50``). The wrapper picks the top-K_t
    long and bottom-K_t short by InVAR score that day, equal-weight, then
    the exposure scalar multiplies the realised L/S return.

    The reward, exposure-change band, vol/drawdown bookkeeping, and
    regime-change counter are byte-for-byte equivalent to the canonical
    1-D env so the Ablation 6 vs canonical delta isolates the concentration
    knob.
    """

    metadata = {"render_modes": []}

    def __init__(self, tape: VariableKTape, cfg, bootstrap_episode: bool = False) -> None:
        from invar_rl.layer3_control.reward import RewardFunction

        super().__init__()
        if len(tape) < 2:
            raise ValueError("tape must contain at least two steps")
        self._tape = tape
        self._cfg = cfg
        self._bootstrap = bootstrap_episode
        self._ep_len = min(cfg.episode_days, len(tape))

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim(tape.episode),),
            dtype=np.float32,
        )
        # 2-D action: [exposure, k_logit].
        self.action_space = spaces.Box(
            low=np.array(
                [float(cfg.exposure_min), -5.0], dtype=np.float32
            ),
            high=np.array(
                [float(cfg.exposure_max), 5.0], dtype=np.float32
            ),
            shape=(2,),
            dtype=np.float32,
        )

        self._reward_fn = RewardFunction(cfg)
        self._risk = RiskState()
        self._start = 0
        self._t = 0
        self._equity = 1.0
        self._hwm = 1.0
        self._ret_hist: List[float] = []
        self._pvol_hist: List[float] = []
        self._k_hist: List[int] = []

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        max_start = len(self._tape) - self._ep_len
        if self._bootstrap and max_start > 0:
            self._start = int(self.np_random.integers(0, max_start + 1))
        else:
            self._start = 0
        self._t = 0
        self._equity = 1.0
        self._hwm = 1.0
        self._ret_hist = []
        self._pvol_hist = []
        self._k_hist = []
        self._reward_fn.reset()
        self._risk = RiskState(
            rolling_vol=0.0,
            drawdown=0.0,
            exposure=float(self._cfg.exposure_min),
            days_since_regime_change=0.0,
        )
        from invar_rl.layer3_control.observation import build_observation
        obs = build_observation(self._tape.episode, self._start, self._risk)
        return obs.astype(np.float32), {}

    def _detect_regime_change(self, pvol: float) -> None:
        self._pvol_hist.append(pvol)
        if len(self._pvol_hist) > _VOL_WINDOW:
            ref = np.asarray(self._pvol_hist[-(_VOL_WINDOW + 1):-1])
            mu, sd = ref.mean(), ref.std()
            if sd > 1e-12 and abs(pvol - mu) > _REGIME_Z * sd:
                self._risk.days_since_regime_change = 0.0
                return
        self._risk.days_since_regime_change += 1.0

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        from invar_rl.layer3_control.observation import build_observation

        idx = self._start + self._t

        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.size != 2:
            raise ValueError(
                f"Ablation 6 expects 2-D action (exposure, k_logit); "
                f"got shape {a.shape}"
            )
        exposure_raw = float(a[0])
        k_logit = float(np.clip(a[1], -5.0, 5.0))

        # Exposure mechanics: same band-limited clip as the canonical env.
        target = float(np.clip(
            exposure_raw, self._cfg.exposure_min, self._cfg.exposure_max
        ))
        prev = self._risk.exposure
        band = self._cfg.exposure_change_band
        exposure = float(np.clip(target, prev - band, prev + band))
        exposure = float(np.clip(
            exposure, self._cfg.exposure_min, self._cfg.exposure_max
        ))

        # Macro-conditional concentration: K_t from the action's sigmoid.
        k_t = _k_from_logit(k_logit)
        long_n = int(self._tape.long_counts[idx])
        short_n = int(self._tape.short_counts[idx])
        k_use = int(min(k_t, long_n, short_n))
        if k_use < _K_MIN:
            k_use = max(1, min(long_n, short_n))
        # Equal-weight L/S over top-k_use long + bottom-k_use short.
        lret = float(self._tape.long_returns[idx, :k_use].mean())
        sret = float(self._tape.short_returns[idx, :k_use].mean())
        base_ret = lret - sret
        strat_ret = exposure * base_ret

        # Traded notional: gross book is 1.0 (sum |w| = 1) so the
        # exposure-delta turnover matches the canonical fixed-wrapper env.
        traded_notional = abs(exposure - prev) * 1.0

        self._equity *= 1.0 + strat_ret
        self._hwm = max(self._hwm, self._equity)
        self._risk.drawdown = (
            0.0 if self._hwm <= 0 else 1.0 - self._equity / self._hwm
        )
        self._ret_hist.append(strat_ret)
        if len(self._ret_hist) >= 2:
            self._risk.rolling_vol = float(
                np.std(self._ret_hist[-_VOL_WINDOW:])
            )
        d_exposure = exposure - prev
        self._risk.exposure = exposure
        self._detect_regime_change(float(self._tape.episode.pred_vol[idx]))
        self._k_hist.append(int(k_use))

        reward = self._reward_fn(
            strategy_return=strat_ret,
            drawdown=self._risk.drawdown,
            delta_exposure=d_exposure,
            traded_notional=traded_notional,
        )

        self._t += 1
        terminated = False
        truncated = self._t >= self._ep_len - 1
        obs_idx = min(self._start + self._t, len(self._tape) - 1)
        obs = build_observation(self._tape.episode, obs_idx, self._risk)
        info = {
            "equity": self._equity,
            "exposure": exposure,
            "strategy_return": strat_ret,
            "k_t": int(k_use),
            "base_return": base_ret,
        }
        return obs.astype(np.float32), reward, terminated, truncated, info


def _rollout(env, agent) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one deterministic rollout; return (returns, exposures, K_t)."""
    obs, _ = env.reset(seed=0)
    rets: List[float] = []
    exps: List[float] = []
    ks: List[int] = []
    while True:
        action, _ = agent.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32)
        obs, _, term, trunc, info = env.step(action)
        rets.append(float(info["strategy_return"]))
        exps.append(float(info["exposure"]))
        ks.append(int(info["k_t"]))
        if term or trunc:
            break
    return (
        np.asarray(rets, dtype=np.float64),
        np.asarray(exps, dtype=np.float64),
        np.asarray(ks, dtype=np.int64),
    )


def _pooled_sharpe(returns: np.ndarray) -> float:
    if returns.size < 2:
        return 0.0
    sd = float(returns.std(ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float(returns.mean() / sd * np.sqrt(_TRADING_DAYS))


class _ValSharpeSelector:
    """SB3 callback: keep the in-memory checkpoint with best val Sharpe."""

    def __init__(self, val_env, eval_freq: int) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        self._val_env = val_env
        self._eval_freq = int(max(1, eval_freq))
        self.best_sharpe = -np.inf
        self.best_step = 0
        self.eval_history: List[Dict[str, float]] = []
        self._buffer: Optional[bytes] = None

        outer = self

        class _Inner(BaseCallback):
            def _on_step(self_inner) -> bool:  # type: ignore[override]
                if self_inner.num_timesteps % outer._eval_freq != 0:
                    return True
                rets, _, ks = _rollout(outer._val_env, self_inner.model)
                sh = _pooled_sharpe(rets)
                outer.eval_history.append({
                    "step": int(self_inner.num_timesteps),
                    "val_sharpe": float(sh),
                    "val_mean": float(rets.mean()) if rets.size else 0.0,
                    "val_std": float(rets.std(ddof=1)) if rets.size > 1 else 0.0,
                    "val_mean_k": float(ks.mean()) if ks.size else 0.0,
                })
                if sh > outer.best_sharpe:
                    outer.best_sharpe = float(sh)
                    outer.best_step = int(self_inner.num_timesteps)
                    buf = io.BytesIO()
                    self_inner.model.save(buf)
                    outer._buffer = buf.getvalue()
                return True

        self.callback = _Inner()

    def restore_best(self, agent) -> bool:
        if self._buffer is None:
            return False
        from stable_baselines3 import SAC

        buf = io.BytesIO(self._buffer)
        env = agent.get_env()
        restored = SAC.load(buf, env=env, device=agent.device)
        agent.policy.load_state_dict(restored.policy.state_dict())
        return True


def _persist_test_outputs(
    tape: VariableKTape,
    rets: np.ndarray,
    exps: np.ndarray,
    ks: np.ndarray,
    bridge,
    out_path: Path,
) -> Dict[str, object]:
    """Write the per-test-day exposures + K_t + realised returns parquet."""
    n = int(min(tape.episode.days.shape[0], rets.shape[0], exps.shape[0], ks.shape[0]))
    if n == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=[
            "date", "exposure", "k_t", "strategy_return", "base_return",
        ]).to_parquet(out_path, index=False)
        return {
            "n_test_days": 0, "sharpe": 0.0, "mean": 0.0, "std": 0.0,
            "mean_k": 0.0, "std_k": 0.0,
        }
    day_indices = tape.episode.days[:n].astype(int)
    base_ret = tape.episode.base_return[:n].astype(np.float64)
    dates = np.asarray([str(bridge.dates[int(d)]) for d in day_indices])
    df = pd.DataFrame({
        "date": pd.to_datetime(dates).normalize(),
        "exposure": exps[:n].astype(np.float64),
        "k_t": ks[:n].astype(np.int64),
        "strategy_return": rets[:n].astype(np.float64),
        "base_return": base_ret,
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return {
        "n_test_days": int(df.shape[0]),
        "sharpe": float(_pooled_sharpe(rets[:n])),
        "mean": float(rets[:n].mean()),
        "std": float(rets[:n].std(ddof=1)) if n > 1 else 0.0,
        "mean_k": float(ks[:n].mean()),
        "std_k": float(ks[:n].std(ddof=1)) if n > 1 else 0.0,
    }


def _build_sac_agent(env, stage3, seed: int):
    """Construct an SB3 SAC agent with the canonical Policy P1 hyperparams.

    Reuses the same hyperparameter triplet as :func:`invar_rl.layer3_control.agent.build_agent`
    (lr=stage3.learning_rate, batch_size=256, twin Q via SB3 default,
    MlpPolicy). Only the env's 2-D action space differs.
    """
    from stable_baselines3 import SAC

    return SAC(
        "MlpPolicy",
        env,
        learning_rate=stage3.learning_rate,
        seed=int(seed),
        verbose=0,
    )


def run_one_cell(
    fold: int,
    seed: int,
    ckpt_path: Path,
    layer3_yaml: Path,
    stage3_yaml: Path,
    total_timesteps: int,
    eval_freq: int,
    output_dir_root: Path,
    panel_end: str,
    panel_kind: str = "lattice_native",
    two_regime_val: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """Train 2-D SAC for one (fold, seed) cell; persist outputs.

    Skips silently if both the per-day parquet and the summary JSON
    already exist (skip-if-output-exists guard).
    """
    out_dir = output_dir_root
    out_path = out_dir / f"fold{fold}_seed{seed}.parquet"
    summary_dir = out_dir / "summary"
    summary_path = summary_dir / f"fold{fold}_seed{seed}.json"
    if out_path.exists() and summary_path.exists():
        print(
            f"[ablation6] {out_path} + summary exist; skipping cell",
            flush=True,
        )
        with open(summary_path) as fh:
            return json.load(fh)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_global_seed(seed)
    layer3 = load_layer3_config(str(layer3_yaml))
    stage3 = load_stage3_config(str(stage3_yaml))

    cfg = InVARConfig(fold=fold, seed=seed)
    cfg.panel_kind = panel_kind
    cfg.two_regime_val = two_regime_val
    cfg.panel_end = panel_end
    cfg.enable_retrieval_bank = False
    bridge = build_lattice_bridge(cfg)
    bundle = load_trained_invar(
        ckpt_path=ckpt_path, bridge=bridge, device=device
    )

    print(
        f"[ablation6] fold={fold} seed={seed} panel_kind={panel_kind} "
        f"device={device} K_range=[{_K_MIN},{_K_MAX}]",
        flush=True,
    )
    t0 = time.time()

    train_tape = precompute_variable_k_tape(
        bundle=bundle, bridge=bridge,
        day_indices=list(bridge.train_idx),
        stride=stage3.precompute_stride,
    )
    val_tape = precompute_variable_k_tape(
        bundle=bundle, bridge=bridge,
        day_indices=list(bridge.val_idx),
        stride=stage3.precompute_stride,
    )
    test_tape = precompute_variable_k_tape(
        bundle=bundle, bridge=bridge,
        day_indices=list(bridge.test_idx),
        stride=stage3.precompute_stride,
    )
    print(
        f"[ablation6] precompute sizes: "
        f"train={len(train_tape)} val={len(val_tape)} test={len(test_tape)} "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )

    from stable_baselines3.common.monitor import Monitor

    train_env = Monitor(
        VariableKExposureEnv(train_tape, layer3, bootstrap_episode=True)
    )
    val_env = VariableKExposureEnv(val_tape, layer3, bootstrap_episode=False)
    test_env = VariableKExposureEnv(test_tape, layer3, bootstrap_episode=False)

    agent = _build_sac_agent(train_env, stage3, seed)

    selector = _ValSharpeSelector(val_env=val_env, eval_freq=eval_freq)
    print(
        f"[ablation6] training 2-D SAC for {total_timesteps:,} steps "
        f"(val every {eval_freq:,})",
        flush=True,
    )
    agent.learn(
        total_timesteps=int(total_timesteps),
        callback=selector.callback,
        progress_bar=False,
    )
    restored = selector.restore_best(agent)
    print(
        f"[ablation6] best val Sharpe={selector.best_sharpe:+.4f} "
        f"at step={selector.best_step} restored={restored}",
        flush=True,
    )

    rets, exps, ks = _rollout(test_env, agent)
    test_stats = _persist_test_outputs(
        tape=test_tape, rets=rets, exps=exps, ks=ks,
        bridge=bridge, out_path=out_path,
    )
    print(
        f"[ablation6] wrote {out_path} "
        f"(test pooled Sharpe={test_stats['sharpe']:+.4f}, "
        f"n_days={test_stats['n_test_days']}, "
        f"mean_K={test_stats['mean_k']:.2f})",
        flush=True,
    )

    summary_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "universe": "sp500",
        "fold": int(fold),
        "seed": int(seed),
        "ablation": "ablation6_macro_conditional_k",
        "model": (
            "InVAR-RL Ablation 6: Layer 1 canonical InVAR + fixed "
            "equal-weight L/S wrapper (variable K via 2-D SAC action) + "
            "Layer 2 SAC. Tests macro-conditional concentration vs fixed K=50."
        ),
        "panel_kind": panel_kind,
        "two_regime_val": bool(two_regime_val),
        "panel_end": panel_end,
        "k_min": int(_K_MIN),
        "k_max": int(_K_MAX),
        "action_space": "(exposure in [0, 1.5], k_logit in [-5, 5]) -> K_t in [5, 50]",
        "n_train_steps": int(len(train_tape)),
        "n_val_steps": int(len(val_tape)),
        "n_test_steps": int(len(test_tape)),
        "total_timesteps": int(total_timesteps),
        "eval_freq": int(eval_freq),
        "best_val_sharpe": float(selector.best_sharpe),
        "best_val_step": int(selector.best_step),
        "best_val_restored": bool(restored),
        "val_history": selector.eval_history,
        "test_pooled_sharpe": float(test_stats["sharpe"]),
        "test_mean_return": float(test_stats["mean"]),
        "test_std_return": float(test_stats["std"]),
        "test_n_days": int(test_stats["n_test_days"]),
        "test_mean_k": float(test_stats["mean_k"]),
        "test_std_k": float(test_stats["std_k"]),
        "test_out_path": str(out_path),
        "layer3_yaml": str(layer3_yaml),
        "stage3_yaml": str(stage3_yaml),
        "wall_time_seconds": float(time.time() - t0),
    }
    with open(summary_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"[ablation6] wrote {summary_path}", flush=True)
    return payload


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "SP500 Ablation 6: macro-conditional concentration via 2-D SAC "
            "action (exposure scalar + K_t in [5, 50])."
        )
    )
    p.add_argument(
        "--fold", type=int, required=True, choices=[1, 2, 3, 4, 5]
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--total-timesteps", type=int, default=20000)
    p.add_argument(
        "--eval-freq", type=int, default=2000,
        help="env steps between val Sharpe checkpoint evaluations",
    )
    p.add_argument(
        "--output-dir-root", type=str,
        default="outputs/sp500/ablations/ablation6",
    )
    p.add_argument(
        "--layer1-ckpt-root", type=str,
        default="invar_rl/results/stage1/_ckpt",
    )
    p.add_argument(
        "--layer3", type=str, default="invar_rl/configs/layer3.yaml"
    )
    p.add_argument(
        "--stage3", type=str, default="invar_rl/configs/stage3.yaml"
    )
    p.add_argument("--panel-end", type=str, default="2025-12-31")
    p.add_argument(
        "--panel-kind", type=str, default="lattice_native",
        choices=["lattice_native", "biotech"],
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    ckpt_path = (
        Path(args.layer1_ckpt_root)
        / f"fold{args.fold}_seed{args.seed}_full.pt"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Layer 1 full ckpt missing for fold={args.fold} "
            f"seed={args.seed}: {ckpt_path}"
        )
    out_root = Path(args.output_dir_root)
    out_path = out_root / f"fold{args.fold}_seed{args.seed}.parquet"
    summary_path = (
        out_root / "summary" / f"fold{args.fold}_seed{args.seed}.json"
    )
    if out_path.exists() and summary_path.exists():
        print(
            f"[ablation6] {out_path} + summary exist; skipping cell",
            flush=True,
        )
        return 0
    run_one_cell(
        fold=args.fold,
        seed=args.seed,
        ckpt_path=ckpt_path,
        layer3_yaml=Path(args.layer3),
        stage3_yaml=Path(args.stage3),
        total_timesteps=int(args.total_timesteps),
        eval_freq=int(args.eval_freq),
        output_dir_root=out_root,
        panel_end=args.panel_end,
        panel_kind=args.panel_kind,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
