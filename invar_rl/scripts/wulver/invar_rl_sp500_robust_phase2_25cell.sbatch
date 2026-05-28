#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# Robust-InVAR-RL Phase 2 25-cell SP500: residual SAC + Kelly prior.
# 5 parallel slots, each handling a round-robin subset of the 25 (fold, seed)
# tuples. Uses Phase 1 group-DRO L1 ckpts as the frozen ranker. Daily tape
# capture is always ON for the paired bootstrap pipeline.
#
# Single sbatch per [[feedback_wulver_single_job]]; generic job name per
# [[feedback_wulver_job_names]].
#
# Submit (5 parallel slots):
#   for SLOT in 0 1 2 3 4; do
#     sbatch --export=ALL,SLOT=$SLOT \
#       invar_rl/scripts/wulver/invar_rl_sp500_robust_phase2_25cell.sbatch
#   done

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

SLOT="${SLOT:?SLOT must be set via --export=SLOT=0..4}"

DELTA_CAP="${DELTA_CAP:-0.25}"
KAPPA="${KAPPA:-0.05}"
E_MAX="${E_MAX:-1.5}"
CALIBRATION="${CALIBRATION:-platt}"
EQUAL_K="${EQUAL_K:-50}"

CKPT_ROOT_PRIMARY="invar_rl/results/stage1_robust_phase1/_ckpt"
CKPT_ROOT_FALLBACK="outputs/sp500/layer1_robust_phase1_25cell/_ckpt"
if [ -d "$CKPT_ROOT_PRIMARY" ] && \
   ls "${CKPT_ROOT_PRIMARY}"/fold*_seed*_full.pt > /dev/null 2>&1; then
  CKPT_ROOT="$CKPT_ROOT_PRIMARY"
elif [ -d "$CKPT_ROOT_FALLBACK" ] && \
     ls "${CKPT_ROOT_FALLBACK}"/fold*_seed*_full.pt > /dev/null 2>&1; then
  CKPT_ROOT="$CKPT_ROOT_FALLBACK"
else
  echo "[ERR] no Phase 1 L1 ckpts found in either path; aborting"
  echo "  tried: ${CKPT_ROOT_PRIMARY}"
  echo "  tried: ${CKPT_ROOT_FALLBACK}"
  exit 2
fi
echo "[INFO] L1 ckpt root: $CKPT_ROOT"

OUT_ROOT="outputs/sp500/layer3_robust_phase2_25cell"
TAPE_ROOT="${OUT_ROOT}/daily_tape"
LOG_ROOT="invar_rl/logs/sp500_robust_phase2_25cell"
mkdir -p "$OUT_ROOT" "$TAPE_ROOT" "$LOG_ROOT"

# 25 (fold, seed) tuples; round-robin over 5 slots so each slot owns 5 cells.
TUPLES=()
for FOLD in 1 2 3 4 5; do
  for SEED in 42 43 44 45 46; do
    TUPLES+=("${FOLD}|${SEED}")
  done
done
N_TUPLES=${#TUPLES[@]}
N_SLOTS=5

echo "host=$(hostname) start=$(date -Is) slot=$SLOT total_tuples=$N_TUPLES"
echo "phase2 25cell: delta_cap=${DELTA_CAP} kappa=${KAPPA} e_max=${E_MAX} cal=${CALIBRATION} K=${EQUAL_K}"
nvidia-smi | head -8 || true

for i in $(seq 0 $((N_TUPLES - 1))); do
  if [ $((i % N_SLOTS)) -ne "$SLOT" ]; then
    continue
  fi
  IFS='|' read -r FOLD SEED <<< "${TUPLES[$i]}"
  CKPT="${CKPT_ROOT}/fold${FOLD}_seed${SEED}_full.pt"
  OUT_JSON="${OUT_ROOT}/fold${FOLD}_seed${SEED}.json"
  LF="${LOG_ROOT}/fold${FOLD}_seed${SEED}.log"

  if [ ! -f "$CKPT" ]; then
    echo "[$(date +%H:%M:%S)] skip fold=$FOLD seed=$SEED: ckpt missing $CKPT"
    continue
  fi
  if [ -f "$OUT_JSON" ]; then
    echo "[$(date +%H:%M:%S)] skip fold=$FOLD seed=$SEED (json present)"
    continue
  fi
  echo "[$(date +%H:%M:%S)] run fold=$FOLD seed=$SEED"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.sp500_residual_sac \
    --fold "$FOLD" --seed "$SEED" \
    --layer1-ckpt "$CKPT" \
    --output-dir "$OUT_ROOT" \
    --daily-tape-dir "$TAPE_ROOT" \
    --panel_kind lattice_native \
    --panel_end 2025-12-31 \
    --two_regime_val \
    --equal-topk-k "$EQUAL_K" \
    --delta-cap "$DELTA_CAP" \
    --kappa "$KAPPA" \
    --e-max "$E_MAX" \
    --calibration-method "$CALIBRATION" \
    --mode e_star_plus_sac \
    > "$LF" 2>&1 || {
      echo "FAILED fold=$FOLD seed=$SEED; see $LF"
      tail -50 "$LF"
      continue
    }
done

echo "DONE slot=${SLOT} $(date -Is)"
ls -la "$OUT_ROOT/" 2>/dev/null || true
