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

# Robust-InVAR-RL Phase 2 smoke: SP500 fold=2 seed=42.
# Runs TWO arms on the same Phase 1 group-DRO L1 ckpt:
#   1. mode=e_star_only    : prior alone, no SAC residual.
#   2. mode=e_star_plus_sac: residual SAC on top of the prior.
# Goal: verify the prior on its own is sane (Sharpe near constant_full
# but vol-scaled) and residual SAC adds value.
#
# Submit:
#   sbatch invar_rl/scripts/wulver/invar_rl_sp500_robust_phase2_smoke.sbatch

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
KAPPA="${KAPPA:-0.05}"
E_MAX="${E_MAX:-1.5}"
CALIBRATION="${CALIBRATION:-platt}"
EQUAL_K="${EQUAL_K:-50}"

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

OUT_PRIOR_DIR="outputs/sp500/layer3_robust_phase2_smoke/e_star_only"
OUT_FULL_DIR="outputs/sp500/layer3_robust_phase2_smoke/e_star_plus_sac"
LOG_DIR="invar_rl/logs/sp500_robust_phase2_smoke"
mkdir -p "$OUT_PRIOR_DIR" "$OUT_FULL_DIR" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is) fold=${FOLD} seed=${SEED}"
echo "phase2 smoke: delta_cap=${DELTA_CAP} kappa=${KAPPA} e_max=${E_MAX} cal=${CALIBRATION} K=${EQUAL_K}"
nvidia-smi | head -8 || true

# Arm 1: prior alone (no SAC residual).
LF1="${LOG_DIR}/fold${FOLD}_seed${SEED}_e_star_only.log"
echo "[$(date +%H:%M:%S)] arm 1: e_star_only"
PYTHONPATH=$PWD python3 -u -m invar_rl.training.sp500_residual_sac \
  --fold "$FOLD" --seed "$SEED" \
  --layer1-ckpt-root "$CKPT_ROOT" \
  --output-dir "$OUT_PRIOR_DIR" \
  --daily-tape-dir "$OUT_PRIOR_DIR/daily_tape" \
  --panel_kind lattice_native \
  --panel_end 2025-12-31 \
  --two_regime_val \
  --equal-topk-k "$EQUAL_K" \
  --delta-cap "$DELTA_CAP" \
  --kappa "$KAPPA" \
  --e-max "$E_MAX" \
  --calibration-method "$CALIBRATION" \
  --mode e_star_only \
  > "$LF1" 2>&1 || {
    echo "FAILED e_star_only; see $LF1"
    tail -50 "$LF1"
    exit 3
  }
echo "[INFO] e_star_only done; summary:"
python3 -c "import json; d=json.load(open('${OUT_PRIOR_DIR}/fold${FOLD}_seed${SEED}.json')); s=d['methods']['sac']; print(f\"  mean_return={s['mean_return']:+.6f}  vol={s['volatility']:.6f}  exp[min/mean/max]={s['exposure_min']:.3f}/{s['exposure_mean']:.3f}/{s['exposure_max']:.3f}\")"

# Arm 2: e_star + SAC residual.
LF2="${LOG_DIR}/fold${FOLD}_seed${SEED}_e_star_plus_sac.log"
echo "[$(date +%H:%M:%S)] arm 2: e_star_plus_sac"
PYTHONPATH=$PWD python3 -u -m invar_rl.training.sp500_residual_sac \
  --fold "$FOLD" --seed "$SEED" \
  --layer1-ckpt-root "$CKPT_ROOT" \
  --output-dir "$OUT_FULL_DIR" \
  --daily-tape-dir "$OUT_FULL_DIR/daily_tape" \
  --panel_kind lattice_native \
  --panel_end 2025-12-31 \
  --two_regime_val \
  --equal-topk-k "$EQUAL_K" \
  --delta-cap "$DELTA_CAP" \
  --kappa "$KAPPA" \
  --e-max "$E_MAX" \
  --calibration-method "$CALIBRATION" \
  --mode e_star_plus_sac \
  > "$LF2" 2>&1 || {
    echo "FAILED e_star_plus_sac; see $LF2"
    tail -50 "$LF2"
    exit 4
  }
echo "[INFO] e_star_plus_sac done; summary:"
python3 -c "import json; d=json.load(open('${OUT_FULL_DIR}/fold${FOLD}_seed${SEED}.json')); s=d['methods']['sac']; print(f\"  mean_return={s['mean_return']:+.6f}  vol={s['volatility']:.6f}  exp[min/mean/max]={s['exposure_min']:.3f}/{s['exposure_mean']:.3f}/{s['exposure_max']:.3f}  residual_abs_mean={s['residual_abs_mean']:.4f}\")"

echo "DONE phase2 smoke $(date -Is)"
