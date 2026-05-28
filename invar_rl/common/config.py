"""YAML to typed dataclass loading with validation.

All hyperparameters, paths, fold definitions, and architectural dimensions
live in YAML and are parsed into typed dataclasses. No magic numbers are
buried in function bodies elsewhere in the project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file into a dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed mapping.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the top-level YAML document is not a mapping.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"top-level YAML in {p} must be a mapping")
    return data


@dataclass(frozen=True)
class SyntheticConfig:
    """Parameters of the synthetic data generator."""

    n_tickers: int
    n_days: int
    n_features: int
    macro_dim: int
    n_regimes: int
    lookback: int
    noise_std: float
    label_horizon: int
    listing_span_frac: float
    inactive_prob: float

    def __post_init__(self) -> None:
        if self.lookback < 1:
            raise ValueError("synthetic.lookback must be >= 1")
        if self.label_horizon < 1:
            raise ValueError("synthetic.label_horizon must be >= 1")
        if self.n_days <= self.lookback + self.label_horizon:
            raise ValueError(
                "synthetic.n_days must exceed lookback plus label_horizon"
            )
        if not 0.0 <= self.inactive_prob < 1.0:
            raise ValueError("synthetic.inactive_prob must be in [0, 1)")


@dataclass(frozen=True)
class PathsConfig:
    """Output locations."""

    output_dir: str
    checkpoint_dir: str
    log_dir: str


@dataclass(frozen=True)
class SP500DataConfig:
    """Real S&P 500 universal-panel data source."""

    data_root: str
    lookback: int
    label_horizon: int
    max_active: int = 0  # 0 = no cap; >0 fixes the daily cross-section size


@dataclass(frozen=True)
class BaseConfig:
    """Global settings shared by every phase."""

    seeds: List[int]
    device: str
    log_level: str
    synthetic: SyntheticConfig
    paths: PathsConfig
    panel_kind: str = "synthetic"
    sp500: Optional[SP500DataConfig] = None

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("base.seeds must be a non-empty list")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("base.seeds must not contain duplicates")
        if self.panel_kind not in ("synthetic", "sp500"):
            raise ValueError(
                "base.panel_kind must be 'synthetic' or 'sp500'"
            )
        if self.panel_kind == "sp500" and self.sp500 is None:
            raise ValueError(
                "base.panel_kind='sp500' requires an sp500 config block"
            )


@dataclass(frozen=True)
class FoldSpec:
    """One walk-forward fold given as inclusive day-index ranges."""

    name: str
    train: Tuple[int, int]
    val: Tuple[int, int]
    test: Tuple[int, int]
    ood: bool = False

    def __post_init__(self) -> None:
        for label, rng in (
            ("train", self.train),
            ("val", self.val),
            ("test", self.test),
        ):
            if len(rng) != 2 or rng[0] > rng[1]:
                raise ValueError(
                    f"fold {self.name}: {label} range must be "
                    f"[start, end] with start <= end, got {rng}"
                )


@dataclass(frozen=True)
class FoldsConfig:
    """Walk-forward fold definitions and the global embargo."""

    embargo_days: int
    folds: List[FoldSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.embargo_days < 0:
            raise ValueError("folds.embargo_days must be >= 0")
        if not self.folds:
            raise ValueError("folds.folds must be a non-empty list")
        ood_count = sum(1 for f in self.folds if f.ood)
        if ood_count != 1:
            raise ValueError(
                "exactly one fold must be flagged as the out-of-distribution "
                f"stress fold, found {ood_count}"
            )


def load_base_config(path: str | Path) -> BaseConfig:
    """Load and validate ``configs/base.yaml``."""
    raw = load_yaml(path)
    synthetic = SyntheticConfig(**raw["synthetic"])
    paths = PathsConfig(**raw["paths"])
    sp500 = (
        SP500DataConfig(**raw["sp500"]) if raw.get("sp500") else None
    )
    return BaseConfig(
        seeds=list(raw["seeds"]),
        device=str(raw["device"]),
        log_level=str(raw["log_level"]),
        synthetic=synthetic,
        paths=paths,
        panel_kind=str(raw.get("panel_kind", "synthetic")),
        sp500=sp500,
    )


def load_folds_config(path: str | Path) -> FoldsConfig:
    """Load and validate ``configs/folds.yaml``."""
    raw = load_yaml(path)
    folds = [
        FoldSpec(
            name=str(item["name"]),
            train=tuple(item["train"]),
            val=tuple(item["val"]),
            test=tuple(item["test"]),
            ood=bool(item.get("ood", False)),
        )
        for item in raw["folds"]
    ]
    return FoldsConfig(embargo_days=int(raw["embargo_days"]), folds=folds)


@dataclass(frozen=True)
class Layer1ModelConfig:
    """InVAR ranker architecture."""

    d_model: int
    encoder_layers: int
    encoder_heads: int
    feedforward: int
    dropout: float
    activation: str
    cross_attention_layers: int
    cross_attention_heads: int
    film_gate_init: float

    def __post_init__(self) -> None:
        if self.d_model % self.encoder_heads != 0:
            raise ValueError("d_model must be divisible by encoder_heads")
        if self.d_model % self.cross_attention_heads != 0:
            raise ValueError(
                "d_model must be divisible by cross_attention_heads"
            )
        if self.activation not in ("gelu", "relu"):
            raise ValueError("activation must be 'gelu' or 'relu'")


@dataclass(frozen=True)
class Layer1TrainingConfig:
    """Stage 1 training schedule."""

    epochs: int
    batch_days_per_step: int
    learning_rate: float
    weight_decay: float
    grad_clip: float
    early_stop_patience: int
    select_on: str


@dataclass(frozen=True)
class Layer1Config:
    """Full Layer 1 configuration (``configs/layer1.yaml``)."""

    model: Layer1ModelConfig
    loss_kind: str
    training: Layer1TrainingConfig

    def __post_init__(self) -> None:
        if self.loss_kind not in ("cross_sectional_mse", "listwise_rank"):
            raise ValueError(
                "loss.kind must be 'cross_sectional_mse' or 'listwise_rank'"
            )


def load_layer1_config(path: str | Path) -> Layer1Config:
    """Load and validate ``configs/layer1.yaml``."""
    raw = load_yaml(path)
    model = Layer1ModelConfig(**raw["model"])
    training = Layer1TrainingConfig(**raw["training"])
    return Layer1Config(
        model=model,
        loss_kind=str(raw["loss"]["kind"]),
        training=training,
    )


@dataclass(frozen=True)
class Layer2Config:
    """Allocation-layer configuration (``configs/layer2.yaml``), flattened."""

    estimator: str
    factor_rank: int
    cov_lookback: int
    risk_aversion: float
    per_name_bound: float
    gross_leverage: float
    topk_k: int
    topk_temperature: float
    topk_temperature_anneal: bool

    def __post_init__(self) -> None:
        if self.estimator not in ("ledoit_wolf", "factor_model"):
            raise ValueError(
                "covariance.estimator must be 'ledoit_wolf' or "
                "'factor_model'"
            )
        if not 0.0 < self.per_name_bound <= 1.0:
            raise ValueError("qp_layer.per_name_bound must be in (0, 1]")
        if self.gross_leverage <= 0.0:
            raise ValueError("qp_layer.gross_leverage must be positive")
        if self.risk_aversion <= 0.0:
            raise ValueError("qp_layer.risk_aversion must be positive")
        if self.topk_temperature <= 0.0:
            raise ValueError("topk_layer.temperature must be positive")
        if self.cov_lookback < 2:
            raise ValueError("covariance.lookback must be >= 2")


def load_layer2_config(path: str | Path) -> Layer2Config:
    """Load and validate ``configs/layer2.yaml``."""
    raw = load_yaml(path)
    cov = raw["covariance"]
    qp = raw["qp_layer"]
    topk = raw["topk_layer"]
    return Layer2Config(
        estimator=str(cov["estimator"]),
        factor_rank=int(cov["factor_rank"]),
        cov_lookback=int(cov["lookback"]),
        risk_aversion=float(qp["risk_aversion"]),
        per_name_bound=float(qp["per_name_bound"]),
        gross_leverage=float(qp["gross_leverage"]),
        topk_k=int(topk["k"]),
        topk_temperature=float(topk["temperature"]),
        topk_temperature_anneal=bool(topk["temperature_anneal"]),
    )


@dataclass(frozen=True)
class Stage2Config:
    """Decision-focused training configuration (``configs/stage2.yaml``)."""

    variant: str
    epochs: int
    learning_rate: float
    weight_decay: float
    grad_clip: float
    variance_penalty: float
    train_day_stride: int
    init_from_stage1: bool
    run_control: bool

    def __post_init__(self) -> None:
        if self.variant not in ("A", "B"):
            raise ValueError(
                "stage2.variant must be 'A' (score head only) or 'B' "
                "(all of Layer 1)"
            )
        if self.train_day_stride < 1:
            raise ValueError("stage2.train_day_stride must be >= 1")
        if self.variance_penalty < 0.0:
            raise ValueError("stage2.variance_penalty must be >= 0")


def load_stage2_config(path: str | Path) -> Stage2Config:
    """Load and validate ``configs/stage2.yaml``."""
    raw = load_yaml(path)
    return Stage2Config(
        variant=str(raw["variant"]),
        epochs=int(raw["epochs"]),
        learning_rate=float(raw["learning_rate"]),
        weight_decay=float(raw["weight_decay"]),
        grad_clip=float(raw["grad_clip"]),
        variance_penalty=float(raw["variance_penalty"]),
        train_day_stride=int(raw["train_day_stride"]),
        init_from_stage1=bool(raw["init_from_stage1"]),
        run_control=bool(raw["run_control"]),
    )


@dataclass(frozen=True)
class Layer3Config:
    """Environment and reward configuration (``configs/layer3.yaml``).

    The agent block is consumed in Phase 5 and is not loaded here.
    """

    episode_days: int
    exposure_min: float
    exposure_max: float
    exposure_change_band: float
    extra_action_knobs: bool
    reward_kind: str
    ds_decay: float
    ds_variance_floor: float
    ds_clip: float
    cvar_level: float
    drawdown_penalty: float
    turnover_penalty: float
    transaction_cost_bps: float

    def __post_init__(self) -> None:
        if self.exposure_min >= self.exposure_max:
            raise ValueError(
                "environment.exposure_min must be < exposure_max"
            )
        if self.exposure_change_band <= 0.0:
            raise ValueError(
                "environment.exposure_change_band must be positive"
            )
        if self.reward_kind not in ("differential_sharpe", "cvar"):
            raise ValueError(
                "reward.kind must be 'differential_sharpe' or 'cvar'"
            )
        if not 0.0 < self.cvar_level < 0.5:
            raise ValueError("reward.cvar_level must be in (0, 0.5)")
        if not 0.0 < self.ds_decay < 1.0:
            raise ValueError("reward.ds_decay must be in (0, 1)")
        if self.ds_variance_floor <= 0.0:
            raise ValueError("reward.ds_variance_floor must be positive")
        if self.ds_clip <= 0.0:
            raise ValueError("reward.ds_clip must be positive")


def load_layer3_config(path: str | Path) -> Layer3Config:
    """Load and validate the environment and reward blocks of layer3.yaml."""
    raw = load_yaml(path)
    env = raw["environment"]
    rew = raw["reward"]
    return Layer3Config(
        episode_days=int(env["episode_days"]),
        exposure_min=float(env["exposure_min"]),
        exposure_max=float(env["exposure_max"]),
        exposure_change_band=float(env["exposure_change_band"]),
        extra_action_knobs=bool(env["extra_action_knobs"]),
        reward_kind=str(rew["kind"]),
        ds_decay=float(rew["ds_decay"]),
        ds_variance_floor=float(rew["ds_variance_floor"]),
        ds_clip=float(rew["ds_clip"]),
        cvar_level=float(rew["cvar_level"]),
        drawdown_penalty=float(rew["drawdown_penalty"]),
        turnover_penalty=float(rew["turnover_penalty"]),
        transaction_cost_bps=float(rew["transaction_cost_bps"]),
    )


@dataclass(frozen=True)
class Stage3Config:
    """RL controller and baseline configuration (``configs/stage3.yaml``)."""

    lower_stack: str
    stage2_variant: str
    methods: List[str]
    precompute_stride: int
    total_timesteps: int
    recurrent_hidden: int
    learning_rate: float
    n_steps: int
    vol_annualised_target: float
    myopic_hidden: int
    myopic_epochs: int
    myopic_learning_rate: float

    def __post_init__(self) -> None:
        if self.lower_stack not in ("stage1", "stage2"):
            raise ValueError(
                "stage3.lower_stack must be 'stage1' or 'stage2'"
            )
        if self.stage2_variant not in ("A", "B"):
            raise ValueError("stage3.stage2_variant must be 'A' or 'B'")
        if not self.methods:
            raise ValueError("stage3.methods must be non-empty")
        if self.precompute_stride < 1:
            raise ValueError("stage3.precompute_stride must be >= 1")
        if self.total_timesteps < 1:
            raise ValueError("stage3.total_timesteps must be >= 1")


def load_stage3_config(path: str | Path) -> Stage3Config:
    """Load and validate ``configs/stage3.yaml``."""
    raw = load_yaml(path)
    rl = raw["rl"]
    return Stage3Config(
        lower_stack=str(raw["lower_stack"]),
        stage2_variant=str(raw["stage2_variant"]),
        methods=[str(m) for m in raw["methods"]],
        precompute_stride=int(raw["precompute_stride"]),
        total_timesteps=int(rl["total_timesteps"]),
        recurrent_hidden=int(rl["recurrent_hidden"]),
        learning_rate=float(rl["learning_rate"]),
        n_steps=int(rl["n_steps"]),
        vol_annualised_target=float(raw["vol_target"]["annualised_target"]),
        myopic_hidden=int(raw["myopic_head"]["hidden"]),
        myopic_epochs=int(raw["myopic_head"]["epochs"]),
        myopic_learning_rate=float(raw["myopic_head"]["learning_rate"]),
    )
