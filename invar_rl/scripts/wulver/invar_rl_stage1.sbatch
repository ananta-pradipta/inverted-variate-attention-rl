#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Persist the full layer-1 state_dict alongside the result JSON so
# InVAR-RL stages 2/3 can load the trained canonical InVAR as a
# frozen scorer. The canonical headline pipeline at
# scripts/wulver/invar_clpretrain.sbatch does NOT set this and keeps
# byte-identical output.
export INVAR_SAVE_FULL_STATE=1

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"

# InVAR-RL Stage 1: train Layer 1 = canonical InVAR (bankless +
# clpretrain). One job per FOLD via --export=FOLD=k, 1..5. The Python
# entry point is invar_rl.training.stage1_rank which delegates to
# src.invar.train_invar; the result is byte-identical to the canonical
# scripts/wulver/invar_clpretrain.sbatch but stashes ckpts and JSONs
# under invar_rl/results/stage1 so InVAR-RL stages 2/3 can find them
# without colliding with the standalone InVAR runs in
# results/invar_clpretrain.
FOLD="${FOLD:?FOLD must be set via --export=FOLD=<1..5>}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-10}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-10}"
OUT_DIR="invar_rl/results/stage1"
LOG_DIR="invar_rl/logs/stage1"
CKPT="${OUT_DIR}/_ckpt/fold${FOLD}_encoder.pt"
mkdir -p "$OUT_DIR" "$OUT_DIR/_ckpt" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is) fold=${FOLD}"
nvidia-smi | head -8 || true

# Stage 1.a: contrastive pretrain ONCE for this fold (seed 42, shared
# across all finetune seeds). Skip if the encoder ckpt already exists.
if [ -f "$CKPT" ]; then
  echo "[$(date +%H:%M:%S)] skip pretrain (ckpt exists: $CKPT)"
else
  PLF="${LOG_DIR}/fold${FOLD}_pretrain.log"
  echo "[$(date +%H:%M:%S)] pretrain fold=$FOLD"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage1_rank \
    --fold "$FOLD" --seed 42 \
    --panel_kind lattice_native \
    --two_regime_val \
    --pretrain_only \
    --pretrain_epochs "$PRETRAIN_EPOCHS" \
    --output_dir "$OUT_DIR" \
    > "$PLF" 2>&1
fi

# Stage 1.b: finetune the 5 seeds against the single fold ckpt.
run_finetune() {
  local SEED="$1"
  local OUT="${OUT_DIR}/fold${FOLD}_seed${SEED}.json"
  if [ -f "$OUT" ] && [ "$(grep -c '"epoch"' "$OUT")" -ge 1 ]; then
    echo "[$(date +%H:%M:%S)] skip $OUT"
    return
  fi
  local LF="${LOG_DIR}/fold${FOLD}_seed${SEED}.log"
  echo "[$(date +%H:%M:%S)] finetune fold=$FOLD seed=$SEED"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage1_rank \
    --fold "$FOLD" --seed "$SEED" \
    --panel_kind lattice_native \
    --two_regime_val \
    --skip_pretrain \
    --finetune_epochs "$FINETUNE_EPOCHS" \
    --output_dir "$OUT_DIR" \
    > "$LF" 2>&1
}

for S in 42 43 44 45 46; do
  run_finetune "$S"
done

echo "DONE fold=${FOLD} $(date -Is)"
