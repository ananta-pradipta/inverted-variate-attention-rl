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

# Option F (SAC curriculum) smoke: SP500 fold=2 seed=42 only.
# Warm-starts from outputs/sac_curriculum/sp500/calm_pretrain/seed42/sac_ckpt.zip.
# Compare resulting Sharpe to canonical SP500 fold=2 seed=42 baseline
# (-0.2259 from outputs/sp500/stage3_rl_ablation/canonical_equal_l2_tape).
#
# Submit:
#   sbatch invar_rl/scripts/wulver/sac_curriculum_smoke.sbatch

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

FOLD=2
SEED=42

OUT_ROOT="outputs/sac_curriculum/sp500/smoke_per_cell_finetune"
TAPE_ROOT="${OUT_ROOT}/daily_tape"
LOG_ROOT="invar_rl/logs/sac_curriculum_smoke"
CKPT_ROOT="invar_rl/results/stage1/_ckpt"
PRETRAIN_ROOT="outputs/sac_curriculum/sp500/calm_pretrain"
mkdir -p "$OUT_ROOT" "$TAPE_ROOT" "$LOG_ROOT"

L1_CKPT="${CKPT_ROOT}/fold${FOLD}_seed${SEED}_full.pt"
WARM_CKPT="${PRETRAIN_ROOT}/seed${SEED}/sac_ckpt.zip"
LF="${LOG_ROOT}/fold${FOLD}_seed${SEED}.log"

if [ ! -f "$L1_CKPT" ]; then
  echo "L1 ckpt missing: $L1_CKPT"
  exit 1
fi
if [ ! -f "$WARM_CKPT" ]; then
  echo "warm-start ckpt missing: $WARM_CKPT"
  echo "Run sac_curriculum_calm_pretrain.sbatch first."
  exit 1
fi

echo "host=$(hostname) start=$(date -Is) smoke fold=$FOLD seed=$SEED"
nvidia-smi | head -8 || true

PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage3_rl_ablation \
  --ablation equal_l2 \
  --fold "$FOLD" --seed "$SEED" \
  --layer1-ckpt "$L1_CKPT" \
  --output-dir "$OUT_ROOT" \
  --panel_kind lattice_native \
  --panel_end 2025-12-31 \
  --two_regime_val \
  --equal-topk-k 50 \
  --methods sac \
  --save-daily-tape \
  --daily-tape-dir "$TAPE_ROOT" \
  --warm-start-ckpt "$WARM_CKPT" \
  --warm-start-timesteps 20000 \
  2>&1 | tee "$LF"

echo "DONE $(date -Is)"
python3 -c "
import json, math
d = json.load(open('${OUT_ROOT}/fold${FOLD}_seed${SEED}.json'))
mr = d['methods']['sac']['mean_return']
vol = d['methods']['sac']['volatility']
sh = mr/vol*math.sqrt(252) if vol > 0 else 0.0
print(f'[Option F smoke] fold=${FOLD} seed=${SEED} mean_return={mr:+.6f} vol={vol:.6f} sharpe={sh:+.4f}')
print(f'  canonical baseline = -0.2259')
print(f'  delta = {sh - (-0.2259):+.4f}')
"
