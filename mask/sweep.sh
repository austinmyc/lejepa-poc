#!/usr/bin/env bash
# Hyperparameter sweep — runs 6 configs sequentially on a single GPU.
# Each run is 30k steps (~500M tokens) — enough to distinguish good vs bad configs.
# Results are logged to W&B and MTEB scores saved under mteb_results/.
#
# Usage:
#   bash mask/sweep.sh               # GPU 0, default settings
#   GPU=1 bash mask/sweep.sh         # pin to a different GPU
#   nohup bash mask/sweep.sh &> sweep.log &   # detach
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/mask/env.sh"
if [ -f .env ]; then set -a; source .env; set +a; fi

# ───────────────── shared config ────────────────────────────────────────────
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

CORPUS="Skylion007/openwebtext"
STEPS=30000
WARMUP=1000
D_MODEL=768; D_PROJ=768; ENC_LAYERS=12; N_HEADS=12
BATCH=128; SEQLEN=128
LR=4e-4
SAVE_EVERY=0    # no periodic saves during sweep — final only
TS="$(date +%Y%m%d_%H%M%S)"
# ─────────────────────────────────────────────────────────────────────────────

run() {
    local NAME="$1"; shift
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Starting: $NAME"
    echo "  Args: $*"
    echo "════════════════════════════════════════════════════════════"
    python mask/train.py \
      --corpus "$CORPUS" --steps "$STEPS" \
      --d-model "$D_MODEL" --d-proj "$D_PROJ" \
      --enc-layers "$ENC_LAYERS" --n-heads "$N_HEADS" \
      --batch-size "$BATCH" --seq-len "$SEQLEN" \
      --lr "$LR" --warmup-steps "$WARMUP" \
      --save-every "$SAVE_EVERY" \
      --wandb --mteb --run-name "$NAME" \
      "$@"
    echo "  Done: $NAME"
}

# ───────────────── sweep configs ─────────────────────────────────────────────
# A: pure MSE, no regularization — baseline ceiling for MSE-only training
run "sweep_${TS}_A_lam0_nonorm"      --lam 0.0    --no-normalize-target

# B: tiny reg + no normalize — MSE-driven with minimal collapse insurance
run "sweep_${TS}_B_lam1e4_nonorm"    --lam 0.0001 --no-normalize-target

# C: medium reg + no normalize
run "sweep_${TS}_C_lam1e3_nonorm"    --lam 0.001  --no-normalize-target

# D: current lam, no normalize — isolates effect of normalize_target
run "sweep_${TS}_D_lam6e3_nonorm"    --lam 0.006  --no-normalize-target

# E: medium reg + normalize + shield encoder (SIGReg stays in proj only)
run "sweep_${TS}_E_lam1e3_norm_shield" --lam 0.001 --sigreg-grad-scale 0.0

# F: medium reg + normalize + full grad (current best guess, shorter repro)
run "sweep_${TS}_F_lam1e3_norm_full"   --lam 0.001

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Sweep complete. Check W&B project 'lejepa' for results."
echo "  MTEB scores: mteb_results/sweep_${TS}_*/"
echo "════════════════════════════════════════════════════════════"
