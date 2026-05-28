#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# F2 listmle: Layer 1 wrapper + Layer 3 SAC eval on the listmle L1 ckpt.
# Single-cell SP500 fold=1 seed=42 smoke. Loads the L1 full state_dict
# from outputs/sp500/layer1_f2_listmle_smoke/_ckpt/fold1_seed42_full.pt
# and runs the wrapper eval + stage3 SAC (K=25 equal-topk L/S). Compare
# Sharpe to reports/sp500/phase5_k25_fair_2026-05-24.md canonical
# (fold1 pooled +0.584, per-cell fold1/seed42 = +0.143).
#
# Submit (after Stage 1 smoke job 1046034 completes):
#   sbatch invar_rl/scripts/wulver/invar_rl_sp500_f2_listmle_l2l3_smoke.sbatch

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

FOLD=1
SEED=42

L1_ROOT="outputs/sp500/layer1_f2_listmle_smoke"
L1_CKPT="${L1_ROOT}/_ckpt/fold${FOLD}_seed${SEED}_full.pt"
OUT_ROOT="outputs/sp500/layer3_f2_listmle_smoke"
WRAPPER_DIR="${OUT_ROOT}/wrapper"
SAC_DIR="${OUT_ROOT}/sac_ls"
LOG_DIR="invar_rl/logs/sp500_f2_listmle_l2l3_smoke"
mkdir -p "$WRAPPER_DIR" "$SAC_DIR" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is) fold=${FOLD} seed=${SEED}"
nvidia-smi | head -8 || true

if [ ! -f "$L1_CKPT" ]; then
  echo "L1 ckpt not found: $L1_CKPT"
  echo "Stage 1 smoke (job 1046034) must complete first."
  exit 1
fi

# (1) Wrapper eval: trained InVAR L1 + top-25 L/S wrapper.
WLF="${LOG_DIR}/wrapper_fold${FOLD}_seed${SEED}.log"
echo "[$(date +%H:%M:%S)] wrapper eval fold=$FOLD seed=$SEED"
PYTHONPATH=$PWD python3 -u -m invar_rl.training.sp500_invar_l1_wrapper_eval \
  --fold "$FOLD" --seed "$SEED" \
  --output-dir-root "$WRAPPER_DIR" \
  --layer1-root "$L1_ROOT" \
  --panel-end 2025-12-31 \
  --two-regime-val \
  > "$WLF" 2>&1 || {
    echo "WRAPPER FAILED; see $WLF"
    tail -50 "$WLF"
    exit 2
  }

# (2) Stage 3 SAC (K=25 equal-topk L/S, the canonical Phase 5 SP500 K=25
# protocol). Uses stage3_rl_ablation --ablation equal_l2.
SLF="${LOG_DIR}/sac_fold${FOLD}_seed${SEED}.log"
echo "[$(date +%H:%M:%S)] stage3 SAC L/S K=25 fold=$FOLD seed=$SEED"
PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage3_rl_ablation \
  --ablation equal_l2 \
  --fold "$FOLD" --seed "$SEED" \
  --layer1-ckpt "$L1_CKPT" \
  --layer2 invar_rl/configs/layer2.yaml \
  --layer3 invar_rl/configs/layer3.yaml \
  --stage3 invar_rl/configs/stage3.yaml \
  --output-dir "$SAC_DIR" \
  --methods sac \
  --panel_kind lattice_native \
  --two_regime_val \
  --equal-topk-k 25 \
  > "$SLF" 2>&1 || {
    echo "SAC FAILED; see $SLF"
    tail -50 "$SLF"
    exit 3
  }

echo "DONE fold=${FOLD} seed=${SEED} $(date -Is)"
