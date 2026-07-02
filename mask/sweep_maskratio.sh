#!/usr/bin/env bash
# Masking ratio sweep — tests whether harder masking tasks force the encoder
# to learn semantic structure from scratch (I-JEPA hypothesis for text).
#
# Two scripts in one: pass GPU=A or GPU=B to run the first or second half.
# Run both halves in parallel on two GPUs:
#   GPU=0 bash mask/sweep_maskratio.sh A &> logs/maskratio_A.log &
#   GPU=1 bash mask/sweep_maskratio.sh B &> logs/maskratio_B.log &
#
# Base config: lam=0.001 (stable from sweep), normalize_target=True, no EMA.
# Varying: mask_ratio (0.15→0.75) x mask_strategy (random, block).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/mask/env.sh"
if [ -f .env ]; then set -a; source .env; set +a; fi

mkdir -p logs

# ───────────────── shared config ────────────────────────────────────────────
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

CORPUS="Skylion007/openwebtext"
STEPS=30000
WARMUP=1000
D_MODEL=768; D_PROJ=768; ENC_LAYERS=12; N_HEADS=12
BATCH=128; SEQLEN=128
LR=4e-4
LAM=0.001       # only stable lam from sweep so far
SAVE_EVERY=0
TS="$(date +%Y%m%d_%H%M%S)"
HALF="${1:-A}"  # A = first 3 runs, B = last 3 runs
# ─────────────────────────────────────────────────────────────────────────────

run() {
    local NAME="$1"; shift
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Starting: $NAME  (GPU=$GPU)"
    echo "  Args: $*"
    echo "════════════════════════════════════════════════════════════"
    python mask/train.py \
      --corpus "$CORPUS" --steps "$STEPS" \
      --d-model "$D_MODEL" --d-proj "$D_PROJ" \
      --enc-layers "$ENC_LAYERS" --n-heads "$N_HEADS" \
      --batch-size "$BATCH" --seq-len "$SEQLEN" \
      --lr "$LR" --warmup-steps "$WARMUP" \
      --lam "$LAM" \
      --save-every "$SAVE_EVERY" \
      --wandb --mteb --run-name "$NAME" \
      "$@"
    echo "  Done: $NAME"
}

if [ "$HALF" = "A" ]; then
    # ── Half A (GPU 0 by default) ────────────────────────────────────────────
    # Baseline: current config — reference point for this sweep
    run "maskratio_${TS}_r15_random"  --mask-ratio 0.15 --mask-strategy random

    # 2x harder: forces encoder to recover more context
    run "maskratio_${TS}_r30_random"  --mask-ratio 0.30 --mask-strategy random

    # I-JEPA territory: random 50%
    run "maskratio_${TS}_r50_random"  --mask-ratio 0.50 --mask-strategy random

else
    # ── Half B (GPU 1 by default) ────────────────────────────────────────────
    # Block masking: contiguous masked region, harder to cheat with local cues
    run "maskratio_${TS}_r30_block"   --mask-ratio 0.30 --mask-strategy block

    # I-JEPA style: large contiguous block (~50% masked)
    run "maskratio_${TS}_r50_block"   --mask-ratio 0.50 --mask-strategy block

    # Maximum pressure: 75% block masking — closest to vision I-JEPA (75%)
    run "maskratio_${TS}_r75_block"   --mask-ratio 0.75 --mask-strategy block

fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Half $HALF complete. MTEB results in mteb_results/ and W&B."
echo "════════════════════════════════════════════════════════════"
