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

# SP500 Phase 5 Layer 3 SAC controller training, K=25 PER SIDE
# equal-weight wrapper. Fair-K comparison vs the four wrapped K=25
# ranker baselines in SP500 Panel A (fairness audit issue #1).
#
# Driver: invar_rl.training.stage3_rl_ablation
#   --ablation equal_l2 sets weighting_mode=equal_topk (the canonical
#     SP500 Phase 5 SAC headline path, which produced L/S +0.945 at
#     K=50 default).
#   --equal-topk-k 25 swaps the hardcoded K=50 default for K=25.
#   --long-only toggles between L/S (no flag) and L/O (flag set).
#     Same precompute pipeline, same SAC hyperparams; only the book
#     design (long-short vs long-only) and per-side K change.
#   --methods sac restricts to SAC only (the K=50 equal_l2 +0.945
#     row was the SAC entry of the rec_ppo/ff_ppo/sac sweep).
#   --output-dir routes each protocol into a distinct directory so
#     L/S and L/O JSONs do not collide on fold/seed.
#
# CLI diff vs canonical invar_rl_stage3_rl_ablation.sbatch
# (ABLATION=equal_l2, FOLD=k):
#   - extra flags    : --equal-topk-k 25
#   - extra flags    : --long-only (only for SLOT 5..9)
#   - --methods sac (was recurrent_ppo,feedforward_ppo,sac)
#   - --output-dir explicit (was defaulted via --output-dir-root /
#     <ablation>)
# Layer-1 ckpts + SAC hyperparams + total timesteps + panel_kind +
# two_regime_val + panel_end + Layer 2/3 yamls are unchanged.
#
# Slot layout (mod 5 = fold index, div 5 = protocol index):
#   SLOT 0..4 -> protocol=ls, folds=1..5
#   SLOT 5..9 -> protocol=lo, folds=1..5
#
# Submit (10 parallel slots):
#   for SLOT in 0 1 2 3 4 5 6 7 8 9; do
#     sbatch --export=ALL,SLOT=$SLOT \
#       invar_rl/scripts/wulver/invar_rl_sp500_layer3_k25.sbatch
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

SLOT="${SLOT:?SLOT must be set via --export=SLOT=0..9}"
FOLDS=(1 2 3 4 5)
PROTOS=(ls lo)
FOLD="${FOLDS[$((SLOT % 5))]}"
PROTO="${PROTOS[$((SLOT / 5))]}"
EQUAL_TOPK_K="${EQUAL_TOPK_K:-25}"
METHODS="${METHODS:-sac}"

CKPT_DIR="invar_rl/results/stage1/_ckpt"
OUT_ROOT="outputs/sp500/layer3_k25"
OUT_DIR="${OUT_ROOT}/${PROTO}"
LOG_DIR="invar_rl/logs/sp500_layer3_k25/${PROTO}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

LONG_ONLY_FLAG=""
if [ "$PROTO" = "lo" ]; then
  LONG_ONLY_FLAG="--long-only"
fi

echo "host=$(hostname) start=$(date -Is) slot=$SLOT fold=$FOLD proto=$PROTO methods=$METHODS"
echo "weighting_mode=equal_topk equal_topk_k=${EQUAL_TOPK_K} long_only_flag='${LONG_ONLY_FLAG}'"
nvidia-smi | head -8 || true

run_seed() {
  local SEED="$1"
  local CKPT="${CKPT_DIR}/fold${FOLD}_seed${SEED}_full.pt"
  local OUT="${OUT_DIR}/fold${FOLD}_seed${SEED}.json"
  if [ ! -f "$CKPT" ]; then
    echo "[$(date +%H:%M:%S)] skip seed=$SEED: ckpt missing $CKPT"
    return
  fi
  if [ -f "$OUT" ]; then
    echo "[$(date +%H:%M:%S)] skip $OUT (already done)"
    return
  fi
  local LF="${LOG_DIR}/fold${FOLD}_seed${SEED}.log"
  echo "[$(date +%H:%M:%S)] sp500-k25 fold=$FOLD seed=$SEED proto=$PROTO"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage3_rl_ablation \
    --ablation equal_l2 \
    --fold "$FOLD" --seed "$SEED" \
    --layer1-ckpt "$CKPT" \
    --layer2 invar_rl/configs/layer2.yaml \
    --layer3 invar_rl/configs/layer3.yaml \
    --stage3 invar_rl/configs/stage3.yaml \
    --output-dir "$OUT_DIR" \
    --methods "$METHODS" \
    --panel_kind lattice_native \
    --two_regime_val \
    --equal-topk-k "$EQUAL_TOPK_K" \
    $LONG_ONLY_FLAG \
    > "$LF" 2>&1 || {
      echo "FAILED fold=$FOLD seed=$SEED proto=$PROTO ; see $LF"
      return
    }
}

for S in 42 43 44 45 46; do
  run_seed "$S"
done

echo "DONE slot=$SLOT fold=$FOLD proto=$PROTO $(date -Is)"
