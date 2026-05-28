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

# InVAR-RL-SIA regime-wired Phase 1: unit tests gate + smoke + 5-cell
# SP500 fold-1 mini-sweep at beta_kl=1e-4 WITH the regime_label channel
# enabled. This is the run after wiring the "I" of SIA so aux_inv fires
# on every minibatch (previously a no-op because regime_lookup=None).
#
# Submit:
#   sbatch invar_rl/scripts/wulver/invar_rl_sia_sp500_regime_minisweep.sbatch

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

TIMESTEPS="${TIMESTEPS:-20000}"
EVAL_FREQ="${EVAL_FREQ:-2000}"
FOLD=1
SMOKE_SEED=42
SMOKE_BETA_KL=1e-3
SWEEP_BETA_KL=1e-4

SMOKE_OUT_ROOT="outputs/sp500/layer2_sia/smoke_regime"
SWEEP_OUT_ROOT="outputs/sp500/layer2_sia/minisweep_regime_beta_1e-4"
LOG_ROOT="invar_rl/logs/sp500_layer2_sia_regime"
CKPT_ROOT="invar_rl/results/stage1/_ckpt"
mkdir -p "$SMOKE_OUT_ROOT" "$SWEEP_OUT_ROOT" "$LOG_ROOT"

echo "host=$(hostname) start=$(date -Is) timesteps=$TIMESTEPS"
nvidia-smi | head -8 || true

# ----------------------------------------------------------------------
# Tests gate: must pass the SIA unit tests + new regime-lookup test
# before training begins.
# ----------------------------------------------------------------------
echo "[$(date +%H:%M:%S)] running SIA unit tests"
PYTHONPATH=$PWD python3 -m pytest -x -q \
    tests/test_sia_actor.py \
    tests/test_sia_aux_loss.py \
    tests/test_sia_gate_learnability.py \
    tests/test_sia_critic.py \
    tests/test_sia_sac_subclass.py \
    tests/test_sia_action_range_parity.py \
    tests/test_sia_regime_lookup.py \
    > "$LOG_ROOT/tests.log" 2>&1 || {
      echo "FAILED SIA unit tests; see $LOG_ROOT/tests.log"
      tail -80 "$LOG_ROOT/tests.log"
      exit 2
    }
echo "tests pass"
tail -5 "$LOG_ROOT/tests.log"

# ----------------------------------------------------------------------
# Smoke: fold 1 seed 42 at beta_kl=1e-3 with --regime-label.
# Sanity-check: aux_inv must be non-zero in the train stats.
# ----------------------------------------------------------------------
SMOKE_CKPT="${CKPT_ROOT}/fold${FOLD}_seed${SMOKE_SEED}_full.pt"
if [ ! -f "$SMOKE_CKPT" ]; then
  echo "missing layer1 ckpt: $SMOKE_CKPT"
  exit 1
fi

SMOKE_LF="${LOG_ROOT}/smoke_fold${FOLD}_seed${SMOKE_SEED}.log"
echo "[$(date +%H:%M:%S)] smoke fold=$FOLD seed=$SMOKE_SEED beta_kl=$SMOKE_BETA_KL regime_label=ON"
PYTHONPATH=$PWD python3 -u -m invar_rl.training.sp500_layer2_sia \
    --fold "$FOLD" --seed "$SMOKE_SEED" \
    --total-timesteps "$TIMESTEPS" \
    --eval-freq "$EVAL_FREQ" \
    --output-dir-root "$SMOKE_OUT_ROOT" \
    --layer1-ckpt-root "$CKPT_ROOT" \
    --layer3 invar_rl/configs/layer3.yaml \
    --stage3 invar_rl/configs/stage3.yaml \
    --latent-dim 16 --beta-kl "$SMOKE_BETA_KL" \
    --lambda-gate 1e-4 --lambda-inv 0.1 \
    --group-source macro_kmeans_8 \
    --regime-label \
    --universe-label sp500 \
    > "$SMOKE_LF" 2>&1 || {
      echo "FAILED smoke; see $SMOKE_LF"
      tail -80 "$SMOKE_LF"
      exit 3
    }
echo "smoke done; tail of log:"
tail -25 "$SMOKE_LF"

SMOKE_SUMMARY="${SMOKE_OUT_ROOT}/summary/fold${FOLD}_seed${SMOKE_SEED}.json"
if [ ! -f "$SMOKE_SUMMARY" ]; then
  echo "smoke summary missing: $SMOKE_SUMMARY"
  exit 4
fi

# Inline parity + aux_inv sanity check.
python3 - <<PY
import json
p = "${SMOKE_SUMMARY}"
d = json.load(open(p))
ts = d.get("sia_train_stats", {})
inv = float(ts.get("aux_inv", float("nan")))
print("[smoke parity + aux_inv sanity]")
print(f"  best_val_sharpe = {d.get('best_val_sharpe', float('nan')):+.4f}")
print(f"  test_sharpe     = {d.get('test_pooled_sharpe', float('nan')):+.4f}")
print(f"  ent_coef        = {ts.get('ent_coef', float('nan')):.5f}")
print(f"  log_prob_mean   = {ts.get('log_prob', float('nan')):+.3f}")
print(f"  exposure_mean   = {ts.get('exposure_mean', float('nan')):+.4f}")
print(f"  exposure_std    = {ts.get('exposure_std', float('nan')):+.4f}")
print(f"  gate_open_frac  = {ts.get('gate_open_fraction', float('nan')):+.4f}")
print(f"  aux_inv         = {inv:.6e}")
print(f"  aux_kl          = {ts.get('aux_kl', float('nan')):.6e}")
print(f"  aux_gate_l1     = {ts.get('aux_gate_l1', float('nan')):.6e}")
if inv <= 0.0:
    print("ERROR: aux_inv is zero or negative; the I of SIA is not firing")
    raise SystemExit(7)
print("aux_inv OK: regime invariance term is firing")
PY

# ----------------------------------------------------------------------
# Mini-sweep: fold 1, seeds 42..46 at beta_kl=1e-4 WITH regime_label.
# ----------------------------------------------------------------------
for SEED in 42 43 44 45 46; do
  CKPT="${CKPT_ROOT}/fold${FOLD}_seed${SEED}_full.pt"
  if [ ! -f "$CKPT" ]; then
    echo "missing layer1 ckpt: $CKPT"
    exit 5
  fi
  LF="${LOG_ROOT}/minisweep_fold${FOLD}_seed${SEED}.log"
  echo "[$(date +%H:%M:%S)] mini-sweep fold=$FOLD seed=$SEED beta_kl=$SWEEP_BETA_KL regime_label=ON"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.sp500_layer2_sia \
      --fold "$FOLD" --seed "$SEED" \
      --total-timesteps "$TIMESTEPS" \
      --eval-freq "$EVAL_FREQ" \
      --output-dir-root "$SWEEP_OUT_ROOT" \
      --layer1-ckpt-root "$CKPT_ROOT" \
      --layer3 invar_rl/configs/layer3.yaml \
      --stage3 invar_rl/configs/stage3.yaml \
      --latent-dim 16 --beta-kl "$SWEEP_BETA_KL" \
      --lambda-gate 1e-4 --lambda-inv 0.1 \
      --group-source macro_kmeans_8 \
      --regime-label \
      --universe-label sp500 \
      > "$LF" 2>&1 || {
        echo "FAILED mini-sweep fold=$FOLD seed=$SEED; see $LF"
        tail -50 "$LF"
        exit 6
      }
  SUM="${SWEEP_OUT_ROOT}/summary/fold${FOLD}_seed${SEED}.json"
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); ts=d.get('sia_train_stats',{}); print(f'  seed=${SEED} test_sharpe={d[\"test_pooled_sharpe\"]:+.4f} aux_inv={ts.get(\"aux_inv\",0.0):.4e}')" "$SUM"
done

echo "[$(date +%H:%M:%S)] all done"
ls -la "$SWEEP_OUT_ROOT/" "$SWEEP_OUT_ROOT/summary/" 2>/dev/null | head -20 || true
