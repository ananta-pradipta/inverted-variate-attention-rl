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

# Generic L2L3 SAC eval at the fixed-K equal-weight L/S wrapper for any
# universe (2026-05-27). Used to run the A3 / A4 cross-universe layer-1
# ckpts AND the canonical layer-1 ckpts through the SAME K=25 fixed
# wrapper via invar_rl.training.stage3_rl_ablation, so the InVAR-RL
# family and the K=25 ranker baselines are compared at a matched K on
# NDX-100 and biotech-NBI (the SP500 K=25 column already exists).
#
# Single sbatch per [[feedback_wulver_single_job]]; 5 parallel slots
# round-robin over 25 cells. Generic job name per
# [[feedback_wulver_job_names]].
#
# Required env:
#   CKPT_ROOT     dir holding fold{F}_seed{S}_full.pt layer-1 ckpts
#   OUT_ROOT      output root for the SAC jsons + daily tape
#   PANEL_KIND    nasdaq100 | biotech_nbi_enriched | lattice_native
# Optional env:
#   EQUAL_TOPK_K  per-side L/S wrapper K (default 25)
#
# Submit (5 parallel slots), e.g. A3 NDX:
#   for SLOT in 0 1 2 3 4; do
#     sbatch --export=ALL,SLOT=$SLOT,PANEL_KIND=nasdaq100,\
# CKPT_ROOT=outputs/nasdaq100/layer1_a3_seq_regime_sector_comove/_ckpt,\
# OUT_ROOT=outputs/nasdaq100/stage3_rl_ablation/a3_seq_regime_sector_comove_k25 \
#       invar_rl/scripts/wulver/invar_rl_universe_l2l3_k25.sbatch
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
CKPT_ROOT="${CKPT_ROOT:?CKPT_ROOT must be set}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT must be set}"
PANEL_KIND="${PANEL_KIND:?PANEL_KIND must be set}"
EQUAL_TOPK_K="${EQUAL_TOPK_K:-25}"
# ABLATION selects the wrapper. "equal_l2" (default) = fixed equal-weight
# top-K wrapper (uses --equal-topk-k). "none" = canonical QP allocator
# (variable weights; --equal-topk-k is not passed). Both keep score_mode
# canonical + full obs.
ABLATION="${ABLATION:-equal_l2}"
KFLAG="--equal-topk-k $EQUAL_TOPK_K"
if [ "$ABLATION" = "none" ]; then
  KFLAG=""
fi
# LONG_ONLY=1 appends --long-only so the precompute tape uses long-only
# weights (no shorts) for the L/O protocol comparison vs the long-only
# ranker baselines. Default 0 = long-short dollar-neutral.
LONG_ONLY="${LONG_ONLY:-0}"
LO_FLAG=""
if [ "$LONG_ONLY" = "1" ]; then
  LO_FLAG="--long-only"
fi

TAPE_ROOT="${OUT_ROOT}/daily_tape"
LOG_ROOT="invar_rl/logs/$(basename "$OUT_ROOT")"
mkdir -p "$OUT_ROOT" "$TAPE_ROOT" "$LOG_ROOT"

TUPLES=()
for FOLD in 1 2 3 4 5; do
  for SEED in 42 43 44 45 46; do
    TUPLES+=("${FOLD}|${SEED}")
  done
done
N_TUPLES=${#TUPLES[@]}
N_SLOTS=5

echo "host=$(hostname) start=$(date -Is) slot=$SLOT panel=$PANEL_KIND K=$EQUAL_TOPK_K ckpt_root=$CKPT_ROOT out=$OUT_ROOT"
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
  echo "[$(date +%H:%M:%S)] run l2l3 ablation=$ABLATION K=$EQUAL_TOPK_K panel=$PANEL_KIND long_only=$LONG_ONLY fold=$FOLD seed=$SEED"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage3_rl_ablation \
    --ablation "$ABLATION" \
    --fold "$FOLD" --seed "$SEED" \
    --layer1-ckpt "$CKPT" \
    --output-dir "$OUT_ROOT" \
    --panel_kind "$PANEL_KIND" \
    --panel_end 2025-12-31 \
    --two_regime_val \
    $KFLAG \
    $LO_FLAG \
    --methods sac \
    --save-daily-tape \
    --daily-tape-dir "$TAPE_ROOT" \
    > "$LF" 2>&1 || {
      echo "FAILED fold=$FOLD seed=$SEED; see $LF"
      tail -50 "$LF"
      continue
    }
done

echo "DONE slot=${SLOT} $(date -Is)"
ls -la "$OUT_ROOT/" 2>/dev/null || true
