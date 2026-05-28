#!/bin/bash
# One-stop submission script for biotech NBI Phases 4, 5, 5.5, 6 in
# dependency order. Run from the repo root on a Wulver login node
# after Phase 3 finalises (all 25 cells of
# outputs/biotech_nbi/layer1/_ckpt/fold*_seed*_full.pt exist).
#
# Stages:
#   1. Phase 4 (Layer 2 QP)        : 5 slots, ~12h wall.
#   2. Phase 5 (Layer 3 SAC)       : 10 slots, ~24h wall, depends on Phase 4.
#   3. Phase 5.5 (baselines)       :
#        - Layer-1 ranker baselines: 5 slots, ~24h wall (independent of L2/L3).
#        - Whole-stack RL baselines: 10 slots, ~24h wall (independent).
#        - Non-learning baselines  : CPU-local (run via python directly,
#                                    not via sbatch).
#   4. Phase 6 (4-ablation matrix) : 10 slots, ~18h wall, depends on Phase 4
#                                    (uses the L2 QP path inside the precompute
#                                    tape).
#
# Dependencies are enforced via --dependency=afterok.
#
# Skip-if-output-exists guards are in every sbatch, so re-submitting is
# safe. Generic job name model-eval everywhere
# ([[feedback_wulver_job_names]]).
#
# Usage:
#   bash invar_rl/scripts/submit_biotech_nbi_full.sh
#
# To submit only a subset, set environment variables:
#   SKIP_P4=1 SKIP_P5=1 bash invar_rl/scripts/submit_biotech_nbi_full.sh

set -euo pipefail

REPO="${REPO:-$HOME/phd-research}"
cd "$REPO"

SBATCH_DIR=invar_rl/scripts/wulver

P4_IDS=()
P5_IDS=()
P55_RANKER_IDS=()
P55_RL_IDS=()
P6_IDS=()

echo "=================================================================="
echo "biotech NBI full submission (Phases 4, 5, 5.5, 6)"
echo "Repo: $REPO"
echo "Date: $(date -Is)"
echo "=================================================================="

# ------------------------------------------------------------------ #
# Phase 4: Layer 2 mean-variance QP (5 slots).
# ------------------------------------------------------------------ #
if [ "${SKIP_P4:-0}" = "1" ]; then
  echo "[skip] Phase 4 (SKIP_P4=1)"
else
  echo
  echo "[Phase 4] submitting Layer 2 QP (5 slots)..."
  for SLOT in 0 1 2 3 4; do
    JID=$(sbatch --parsable \
      --export=ALL,SLOT=$SLOT \
      "$SBATCH_DIR/invar_rl_biotech_nbi_layer2.sbatch")
    echo "  Phase 4 slot=$SLOT jobid=$JID"
    P4_IDS+=("$JID")
  done
fi

# Build the dependency-after string for Phase 4 (used by Phase 5 + 6).
P4_DEP=""
if [ ${#P4_IDS[@]} -gt 0 ]; then
  P4_DEP="--dependency=afterok:$(IFS=:; echo "${P4_IDS[*]}")"
fi

# ------------------------------------------------------------------ #
# Phase 5: Layer 3 SAC (10 slots, depends on Phase 4).
# ------------------------------------------------------------------ #
if [ "${SKIP_P5:-0}" = "1" ]; then
  echo "[skip] Phase 5 (SKIP_P5=1)"
else
  echo
  echo "[Phase 5] submitting Layer 3 SAC (10 slots, $P4_DEP)..."
  for SLOT in 0 1 2 3 4 5 6 7 8 9; do
    JID=$(sbatch --parsable $P4_DEP \
      --export=ALL,SLOT=$SLOT \
      "$SBATCH_DIR/invar_rl_biotech_nbi_layer3.sbatch")
    echo "  Phase 5 slot=$SLOT jobid=$JID"
    P5_IDS+=("$JID")
  done
fi

# ------------------------------------------------------------------ #
# Phase 5.5 (independent of Phases 4/5; only needs Phase 3 ckpts +
# raw data, both already on disk after Phase 3).
# ------------------------------------------------------------------ #
if [ "${SKIP_P55:-0}" = "1" ]; then
  echo "[skip] Phase 5.5 (SKIP_P55=1)"
else
  echo
  echo "[Phase 5.5] submitting Layer-1 ranker baselines (5 slots)..."
  for SLOT in 0 1 2 3 4; do
    JID=$(sbatch --parsable \
      --export=ALL,SLOT=$SLOT,N_SLOTS=5 \
      "$SBATCH_DIR/invar_rl_biotech_nbi_baselines_layer1.sbatch")
    echo "  Phase 5.5 ranker slot=$SLOT jobid=$JID"
    P55_RANKER_IDS+=("$JID")
  done

  echo
  echo "[Phase 5.5] submitting whole-stack RL baselines (10 slots)..."
  for SLOT in 0 1 2 3 4 5 6 7 8 9; do
    JID=$(sbatch --parsable \
      --export=ALL,SLOT=$SLOT,N_SLOTS=10 \
      "$SBATCH_DIR/invar_rl_biotech_nbi_baselines_rl.sbatch")
    echo "  Phase 5.5 RL slot=$SLOT jobid=$JID"
    P55_RL_IDS+=("$JID")
  done

  echo
  echo "[Phase 5.5] running non-learning baselines (CPU, local)..."
  echo "  (run on a login or interactive CPU node; takes <60s)"
  echo "  PYTHONPATH=$PWD python3 invar_rl/scripts/biotech_nbi_non_learning_baselines.py"
fi

# ------------------------------------------------------------------ #
# Phase 6: 4-ablation matrix (10 slots, depends on Phase 4).
# ------------------------------------------------------------------ #
if [ "${SKIP_P6:-0}" = "1" ]; then
  echo "[skip] Phase 6 (SKIP_P6=1)"
else
  echo
  echo "[Phase 6] submitting 4-ablation matrix (10 slots, $P4_DEP)..."
  for SLOT in 0 1 2 3 4 5 6 7 8 9; do
    JID=$(sbatch --parsable $P4_DEP \
      --export=ALL,SLOT=$SLOT \
      "$SBATCH_DIR/invar_rl_biotech_nbi_phase6_ablation.sbatch")
    echo "  Phase 6 slot=$SLOT jobid=$JID"
    P6_IDS+=("$JID")
  done
fi

echo
echo "=================================================================="
echo "Submission complete."
echo "  Phase 4    job ids: ${P4_IDS[*]:-(none)}"
echo "  Phase 5    job ids: ${P5_IDS[*]:-(none)}"
echo "  Phase 5.5  ranker:  ${P55_RANKER_IDS[*]:-(none)}"
echo "  Phase 5.5  RL:      ${P55_RL_IDS[*]:-(none)}"
echo "  Phase 6    job ids: ${P6_IDS[*]:-(none)}"
echo
echo "Rollup commands (run after each phase completes):"
echo "  Phase 4 health:     PYTHONPATH=\$PWD python3 -c 'import json; "
echo "                       from pathlib import Path; "
echo "                       [print(p, json.load(open(p))['per_protocol'].keys()) "
echo "                        for p in Path(\"outputs/biotech_nbi/layer2/summary\").glob(\"*.json\")]'"
echo "  Phase 5 rollup:     PYTHONPATH=\$PWD python3 -m invar_rl.scripts.rollup_biotech_nbi_layer3"
echo "  Phase 5.5 rollup:   PYTHONPATH=\$PWD python3 invar_rl/scripts/rollup_biotech_nbi_baselines.py"
echo "  Phase 6 rollup:     PYTHONPATH=\$PWD python3 -m invar_rl.scripts.rollup_biotech_nbi_ablations"
echo "=================================================================="
