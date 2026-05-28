#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# Robust-InVAR-RL Phase 3 smoke: SP500 fold=2 seed=42.
# Builds on top of Phase 2 (Kelly prior + residual SAC) with TWO swaps:
#   1. Compact observation (17 dims: 8 base + 8 regime one-hot + 1 e_star)
#   2. Online Sharpe reward (EWMA Sharpe increment, half-life 21 days)
# The frozen Phase 1 group-DRO L1 ckpt is unchanged; the Kelly prior +
# residual SAC architecture is unchanged; only the obs builder and the
# reward function are swapped.
#
# Expected compact obs dim = 17 (8 base + 8 regime + 1 e_star).
# Phase 2 single-cell smoke reference Sharpe = +0.30 (fold=2 seed=42).
# Phase 3 stop gate (smoke arm): Sharpe within +/- 0.20 of +0.30.
#
# Submit:
#   sbatch invar_rl/scripts/wulver/invar_rl_sp500_robust_phase3_smoke.sbatch

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"

FOLD="${FOLD:-2}"
SEED="${SEED:-42}"
DELTA_CAP="${DELTA_CAP:-0.25}"
# Match Phase 2 25-cell kappa (audit-fix lowered from 0.05 to 0.02).
KAPPA="${KAPPA:-0.02}"
E_MAX="${E_MAX:-1.5}"
CALIBRATION="${CALIBRATION:-platt}"
EQUAL_K="${EQUAL_K:-50}"

# Spec count for the Phase 3 stop gate (obs dim must equal this).
EXPECTED_OBS_DIM=17

CKPT_ROOT_PRIMARY="invar_rl/results/stage1_robust_phase1/_ckpt"
CKPT_ROOT_FALLBACK="outputs/sp500/layer1_robust_phase1_25cell/_ckpt"
if [ -f "${CKPT_ROOT_PRIMARY}/fold${FOLD}_seed${SEED}_full.pt" ]; then
  CKPT_ROOT="$CKPT_ROOT_PRIMARY"
elif [ -f "${CKPT_ROOT_FALLBACK}/fold${FOLD}_seed${SEED}_full.pt" ]; then
  CKPT_ROOT="$CKPT_ROOT_FALLBACK"
else
  echo "[ERR] no Phase 1 L1 ckpt at fold=$FOLD seed=$SEED in either path; aborting"
  echo "  tried: ${CKPT_ROOT_PRIMARY}/fold${FOLD}_seed${SEED}_full.pt"
  echo "  tried: ${CKPT_ROOT_FALLBACK}/fold${FOLD}_seed${SEED}_full.pt"
  exit 2
fi
echo "[INFO] L1 ckpt root: $CKPT_ROOT"

OUT_DIR="outputs/sp500/layer3_robust_phase3_smoke"
LOG_DIR="invar_rl/logs/sp500_robust_phase3_smoke"
mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is) fold=${FOLD} seed=${SEED}"
echo "phase3 smoke: delta_cap=${DELTA_CAP} kappa=${KAPPA} e_max=${E_MAX} cal=${CALIBRATION} K=${EQUAL_K}"
echo "phase3 expected compact obs dim = ${EXPECTED_OBS_DIM} (8 base + 8 regime + 1 e_star)"
nvidia-smi | head -8 || true

LF="${LOG_DIR}/fold${FOLD}_seed${SEED}.log"
echo "[$(date +%H:%M:%S)] phase3 e_star_plus_sac with compact obs + online Sharpe reward"
PYTHONPATH=$PWD python3 -u -m invar_rl.training.sp500_residual_sac \
  --fold "$FOLD" --seed "$SEED" \
  --layer1-ckpt-root "$CKPT_ROOT" \
  --output-dir "$OUT_DIR" \
  --daily-tape-dir "$OUT_DIR/daily_tape" \
  --panel_kind lattice_native \
  --panel_end 2025-12-31 \
  --two_regime_val \
  --equal-topk-k "$EQUAL_K" \
  --delta-cap "$DELTA_CAP" \
  --kappa "$KAPPA" \
  --e-max "$E_MAX" \
  --calibration-method "$CALIBRATION" \
  --mode e_star_plus_sac \
  --phase3-compact-obs \
  --phase3-online-sharpe-reward \
  --phase3-universe-label sp500 \
  --phase3-sharpe-half-life 21 \
  --phase3-sharpe-warmup-steps 5 \
  --phase3-sharpe-clip 8.0 \
  --phase3-vix-scale 20.0 \
  --phase3-ust10y-scale 4.0 \
  > "$LF" 2>&1 || {
    echo "FAILED phase3 smoke; see $LF"
    tail -80 "$LF"
    exit 3
  }

echo "[INFO] phase3 smoke done; summary:"
python3 -c "
import json
d = json.load(open('${OUT_DIR}/fold${FOLD}_seed${SEED}.json'))
s = d['methods']['sac']
p3 = d['diagnostics'].get('phase3', {})
mr = s['mean_return']
vol = s['volatility']
sharpe = (mr / vol * (252 ** 0.5)) if vol > 0 else 0.0
print(f\"  mean_return={mr:+.6f}  vol={vol:.6f}  ann_sharpe={sharpe:+.3f}\")
print(f\"  exp[min/mean/max]={s['exposure_min']:.3f}/{s['exposure_mean']:.3f}/{s['exposure_max']:.3f}\")
print(f\"  residual_abs_mean={s['residual_abs_mean']:.4f}\")
print(f\"  phase3 obs dim: actual={p3.get('compact_obs_dim_actual')} expected={p3.get('compact_obs_dim_expected')} (spec=${EXPECTED_OBS_DIM})\")
print(f\"  phase3 enabled: compact_obs={p3.get('enabled_compact_obs')} online_sharpe={p3.get('enabled_online_sharpe_reward')} regime_one_hot={p3.get('enabled_regime_one_hot')}\")
print(f\"  phase3 regime miss-rate: train={p3.get('regime_miss_rate_train', 0):.3f} test={p3.get('regime_miss_rate_test', 0):.3f}\")
print(f\"  cap_violations={d['diagnostics']['cap_violations']}\")
"

echo "DONE phase3 smoke $(date -Is)"
