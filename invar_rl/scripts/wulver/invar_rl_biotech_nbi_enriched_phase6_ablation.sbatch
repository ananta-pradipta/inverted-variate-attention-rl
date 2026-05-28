#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# Phase 3 of fairness audit issue #11 fix: re-run the biotech NBI
# Phase 6 four-ablation matrix on the ENRICHED 22-feature panel.
# ONE sbatch slot serially processes a fixed shard of the (ablation,
# method, protocol, fold, seed) grid; 10 parallel slots span the whole
# grid. Single-job-per-sbatch per [[feedback_wulver_single_job]];
# generic job name per [[feedback_wulver_job_names]].
#
# Grid:
#   ABLATIONS  = canonical, random_l1, equal_l2, stripped_l3
#   METHODS    = recurrent_ppo, feedforward_ppo, sac, constant_full
#   PROTOCOLS  = ls, lo
#   FOLDS      = 1..5
#   SEEDS      = 42, 43, 44, 45, 46
#
# stripped_l3 + constant_full is undefined; skipped at tuple level.
# 30 valid tuples x 25 cells = 750 cells.
#
# Sharding: each slot picks one of 10 (ablation, method, protocol)
# tuples and runs the full 25-cell fold x seed sweep for it. With
# 30 valid tuples and 10 slots, each slot handles 3 tuples.
#
# Submit (10 parallel slots):
#   for SLOT in 0 1 2 3 4 5 6 7 8 9; do
#     sbatch --export=ALL,SLOT=$SLOT \
#       invar_rl/scripts/wulver/invar_rl_biotech_nbi_enriched_phase6_ablation.sbatch
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
TIMESTEPS="${TIMESTEPS:-20000}"
EVAL_FREQ="${EVAL_FREQ:-2000}"

OUT_ROOT="outputs/biotech_nbi_enriched/phase6_ablation"
LOG_ROOT="invar_rl/logs/biotech_nbi_enriched_phase6_ablation"
CKPT_ROOT="outputs/biotech_nbi_enriched/layer1/_ckpt"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

ABLATIONS=(canonical random_l1 equal_l2 stripped_l3)
METHODS=(recurrent_ppo feedforward_ppo sac constant_full)
PROTOCOLS=(ls lo)

TUPLES=()
for A in "${ABLATIONS[@]}"; do
  for M in "${METHODS[@]}"; do
    if [ "$A" = "stripped_l3" ] && [ "$M" = "constant_full" ]; then
      continue
    fi
    for P in "${PROTOCOLS[@]}"; do
      TUPLES+=("${A}|${M}|${P}")
    done
  done
done
N_TUPLES=${#TUPLES[@]}    # 30
N_SLOTS=10

echo "host=$(hostname) start=$(date -Is) slot=$SLOT total_tuples=$N_TUPLES"
nvidia-smi | head -8 || true

for i in $(seq 0 $((N_TUPLES - 1))); do
  if [ $((i % N_SLOTS)) -ne "$SLOT" ]; then
    continue
  fi
  IFS='|' read -r ABLATION METHOD PROTOCOL <<< "${TUPLES[$i]}"
  echo "[$(date +%H:%M:%S)] tuple $i: ablation=$ABLATION method=$METHOD protocol=$PROTOCOL"

  for FOLD in 1 2 3 4 5; do
    for SEED in 42 43 44 45 46; do
      CKPT="${CKPT_ROOT}/fold${FOLD}_seed${SEED}_full.pt"
      OUT="${OUT_ROOT}/${ABLATION}/${METHOD}/${PROTOCOL}/fold${FOLD}_seed${SEED}.parquet"
      SUMMARY="${OUT_ROOT}/${ABLATION}/${METHOD}/${PROTOCOL}/summary/fold${FOLD}_seed${SEED}.json"
      LOG_DIR="${LOG_ROOT}/${ABLATION}/${METHOD}/${PROTOCOL}"
      mkdir -p "$LOG_DIR"
      LF="${LOG_DIR}/fold${FOLD}_seed${SEED}.log"

      if [ ! -f "$CKPT" ]; then
        echo "[$(date +%H:%M:%S)] skip fold=$FOLD seed=$SEED: ckpt missing $CKPT"
        continue
      fi
      if [ -f "$OUT" ] && [ -f "$SUMMARY" ]; then
        echo "[$(date +%H:%M:%S)] skip ${ABLATION}/${METHOD}/${PROTOCOL} fold=$FOLD seed=$SEED (outputs exist)"
        continue
      fi
      echo "[$(date +%H:%M:%S)] run ${ABLATION}/${METHOD}/${PROTOCOL} fold=$FOLD seed=$SEED"
      PYTHONPATH=$PWD python3 -u -m invar_rl.training.biotech_nbi_ablation_eval \
        --ablation "$ABLATION" \
        --method "$METHOD" \
        --fold "$FOLD" --seed "$SEED" \
        --protocol "$PROTOCOL" \
        --panel-kind biotech_nbi_enriched \
        --total-timesteps "$TIMESTEPS" \
        --eval-freq "$EVAL_FREQ" \
        --output-dir-root "$OUT_ROOT" \
        --layer1-ckpt-root "$CKPT_ROOT" \
        --layer3 invar_rl/configs/layer3.yaml \
        --stage3 invar_rl/configs/stage3.yaml \
        > "$LF" 2>&1 || {
          echo "FAILED ${ABLATION}/${METHOD}/${PROTOCOL} fold=$FOLD seed=$SEED; see $LF"
          continue
        }
    done
  done
done

echo "DONE slot=$SLOT $(date -Is)"
