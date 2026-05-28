#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# InVAR-RL-Q Phase 3 audit-fix smoke: SP500 fold=1 seed=42 with
# actor_hidden=[256, 256] (matching canonical SAC actor capacity).
# Phase 1+2 used default [64, 64] which is 4x narrower than canonical;
# the audit flagged this as a confound. P0 fix per the audit.

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
TIMESTEPS=20000
OUT_ROOT="outputs/sp500/layer2_q/phase3_audit_smoke"
LOG_ROOT="invar_rl/logs/sp500_layer2_q_phase3_audit_smoke"
CKPT_ROOT="invar_rl/results/stage1/_ckpt"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

echo "host=$(hostname) start=$(date -Is) actor_hidden=256 256"
nvidia-smi | head -8 || true

LF="${LOG_ROOT}/fold${FOLD}_seed${SEED}.log"
echo "[$(date +%H:%M:%S)] phase3_audit_smoke fold=$FOLD seed=$SEED"
PYTHONPATH=$PWD python3 -u -m invar_rl.training.sp500_layer2_q \
    --fold "$FOLD" --seed "$SEED" \
    --total-timesteps "$TIMESTEPS" \
    --output-dir-root "$OUT_ROOT" \
    --layer1-ckpt-root "$CKPT_ROOT" \
    --layer3 invar_rl/configs/layer3.yaml \
    --stage3 invar_rl/configs/stage3.yaml \
    --n-quantiles 51 --alpha-cvar 0.1 --eta-blend 0.5 \
    --actor-hidden 256 256 \
    > "$LF" 2>&1 || {
      echo "FAILED fold=$FOLD seed=$SEED; see $LF"
      tail -50 "$LF"
      exit 6
    }

echo "DONE $(date -Is)"
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(f'fold=${FOLD} seed=${SEED} test_sharpe={d[\"test_pooled_sharpe\"]:+.4f}')" "${OUT_ROOT}/summary/fold${FOLD}_seed${SEED}.json"
