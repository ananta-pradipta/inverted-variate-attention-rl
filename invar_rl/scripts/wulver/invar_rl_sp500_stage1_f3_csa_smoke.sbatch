#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# F3 fundamental L1 upgrade smoke (2026-05-26): cross-stock attention.
# Single (fold=1, seed=42) cell on the SP500 lattice_native panel under
# the two-regime val protocol. Adds an explicit cross-stock self-
# attention block at the head boundary, after day-memory fusion and
# before the score head. The canonical path (cross-stock attention off)
# is byte-identical when the flag is not set.
#
# Generic job name model-eval per [[feedback_wulver_job_names]]. One
# sbatch with internal sequential pretrain -> finetune steps per
# [[feedback_wulver_single_job]] (no Slurm array).
#
# Submit:
#   sbatch invar_rl/scripts/wulver/invar_rl_sp500_stage1_f3_csa_smoke.sbatch

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Persist the full layer-1 state_dict so the wrapper / SAC eval can
# load it. The save block in train_invar_clpretrain_v2 writes the F3
# flags into the blob; canonical_runner reads them back.
export INVAR_SAVE_FULL_STATE=1

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"

FOLD=1
SEED=42
PRETRAIN_EPOCHS=10
FINETUNE_EPOCHS=10

OUT_DIR="invar_rl/results/layer1_f3_csa_smoke"
LOG_DIR="invar_rl/logs/layer1_f3_csa_smoke"
CKPT_DIR="${OUT_DIR}/_ckpt"
mkdir -p "$OUT_DIR" "$CKPT_DIR" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is) fold=${FOLD} seed=${SEED}"
nvidia-smi | head -8 || true

# Stage 1.a: per-fold encoder pretrain. The temporal-encoder pretrain
# is INDEPENDENT of the F3 flag (CSA block is downstream of the
# encoder; the pretrainer only trains PerTickerTemporalEncoder
# weights). Same --cross-stock-attn flag passed through for
# consistency; it is ignored during pretrain.
ENC_CKPT="${CKPT_DIR}/fold${FOLD}_encoder.pt"
if [ ! -f "$ENC_CKPT" ]; then
  PLF="${LOG_DIR}/fold${FOLD}_pretrain.log"
  echo "[$(date +%H:%M:%S)] pretrain fold=$FOLD"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage1_rank \
    --fold "$FOLD" --seed "$SEED" \
    --panel_kind lattice_native \
    --panel_end 2025-12-31 \
    --two_regime_val \
    --pretrain_only \
    --pretrain_epochs "$PRETRAIN_EPOCHS" \
    --output_dir "$OUT_DIR" \
    --cross-stock-attn \
    > "$PLF" 2>&1
else
  echo "[$(date +%H:%M:%S)] pretrain ckpt exists; skip pretrain"
fi

# Stage 1.b: finetune the one seed (42) with CSA on.
LF="${LOG_DIR}/fold${FOLD}_seed${SEED}.log"
echo "[$(date +%H:%M:%S)] finetune fold=$FOLD seed=$SEED with --cross-stock-attn"
PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage1_rank \
  --fold "$FOLD" --seed "$SEED" \
  --panel_kind lattice_native \
  --panel_end 2025-12-31 \
  --two_regime_val \
  --skip_pretrain \
  --finetune_epochs "$FINETUNE_EPOCHS" \
  --output_dir "$OUT_DIR" \
  --cross-stock-attn \
  > "$LF" 2>&1

# Brief result rollup so the .out file shows the headline numbers.
python3 -c "
import json
from pathlib import Path
out_path = Path('${OUT_DIR}') / 'fold${FOLD}_seed${SEED}.json'
if not out_path.exists():
    print(f'[F3 smoke] no result JSON at {out_path}')
else:
    d = json.loads(out_path.read_text())
    print('[F3 smoke] FOLD=1 SEED=42 lattice_native two_regime_val')
    print(f'  test ic        = {d.get(\"ic\"):+.4f}')
    print(f'  test rank_ic   = {d.get(\"rank_ic\"):+.4f}')
    print(f'  test ndcg10    = {d.get(\"ndcg10\"):.4f}')
    print(f'  val  ic        = {d.get(\"val_ic\"):+.4f}')
    print(f'  val  rank_ic   = {d.get(\"val_rank_ic\"):+.4f}')
    print('[canonical ref] fold1 seed42 (no CSA): ic=+0.0330 rank_ic=+0.0426')
"

echo "DONE fold=${FOLD} seed=${SEED} $(date -Is)"
