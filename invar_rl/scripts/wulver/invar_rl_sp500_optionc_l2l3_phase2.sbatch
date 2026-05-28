#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# Option C (F2 ListMLE soft + F3 Cross-stock attention compose) Phase 2
# L2L3 evaluation on SP500. 5 slots, one fold per slot, 5 seeds (42..46)
# serial per slot. For each (fold, seed) cell:
#   (1) wrapper eval = trained Option C L1 + top-25 L/S wrapper.
#   (2) stage3 SAC = K=25 equal-topk L/S (the canonical SP500 K=25
#       Phase 5 protocol, ablation=equal_l2 --equal-topk-k 25).
# Uses the Option C Phase 2 L1 ckpts at
# outputs/sp500/layer1_optionc_phase2/.
#
# Generic job name model-eval per [[feedback_wulver_job_names]]. One
# sbatch with internal seed loop per [[feedback_wulver_single_job]].
#
# Submit (one job per fold, 5 jobs total). Run only after the Stage 1
# Option C ckpts have all landed:
#   for F in 1 2 3 4 5; do
#     sbatch --export=ALL,FOLD=$F \
#       invar_rl/scripts/wulver/invar_rl_sp500_optionc_l2l3_phase2.sbatch
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

FOLD="${FOLD:?FOLD must be set via --export=FOLD=<1..5>}"

L1_ROOT="outputs/sp500/layer1_optionc_phase2"
L1_CKPT_DIR="${L1_ROOT}/_ckpt"
OUT_ROOT="outputs/sp500/layer3_optionc_phase2"
WRAPPER_DIR="${OUT_ROOT}/wrapper"
SAC_DIR="${OUT_ROOT}/sac_ls"
LOG_DIR="invar_rl/logs/sp500_optionc_l2l3_phase2"
mkdir -p "$WRAPPER_DIR" "$SAC_DIR" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is) fold=${FOLD}"
nvidia-smi | head -8 || true

run_cell() {
  local SEED="$1"
  local L1_CKPT="${L1_CKPT_DIR}/fold${FOLD}_seed${SEED}_full.pt"
  if [ ! -f "$L1_CKPT" ]; then
    echo "[$(date +%H:%M:%S)] skip seed=$SEED: L1 ckpt missing $L1_CKPT"
    return
  fi

  # (1) Wrapper eval at K=25 (the default _TOP_K_LS_NATIVE in the
  # SP500 wrapper).
  local WRAP_OUT="${WRAPPER_DIR}/invar_l1/fold${FOLD}_seed${SEED}.json"
  if [ -f "$WRAP_OUT" ]; then
    echo "[$(date +%H:%M:%S)] skip wrapper fold=$FOLD seed=$SEED (json present)"
  else
    local WLF="${LOG_DIR}/wrapper_fold${FOLD}_seed${SEED}.log"
    echo "[$(date +%H:%M:%S)] wrapper eval fold=$FOLD seed=$SEED"
    PYTHONPATH=$PWD python3 -u -m invar_rl.training.sp500_invar_l1_wrapper_eval \
      --fold "$FOLD" --seed "$SEED" \
      --output-dir-root "$WRAPPER_DIR" \
      --layer1-root "$L1_ROOT" \
      --panel-end 2025-12-31 \
      --two-regime-val \
      > "$WLF" 2>&1 || {
        echo "WRAPPER FAILED fold=$FOLD seed=$SEED ; see $WLF"
        tail -50 "$WLF"
      }
  fi

  # (2) Stage 3 SAC, K=25 equal-topk L/S (canonical fair-K protocol).
  local SAC_OUT="${SAC_DIR}/fold${FOLD}_seed${SEED}.json"
  if [ -f "$SAC_OUT" ]; then
    echo "[$(date +%H:%M:%S)] skip SAC fold=$FOLD seed=$SEED (json present)"
  else
    local SLF="${LOG_DIR}/sac_fold${FOLD}_seed${SEED}.log"
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
        echo "SAC FAILED fold=$FOLD seed=$SEED ; see $SLF"
        tail -50 "$SLF"
      }
  fi
}

for S in 42 43 44 45 46; do
  run_cell "$S"
done

echo "DONE fold=${FOLD} $(date -Is)"
