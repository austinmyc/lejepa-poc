#!/usr/bin/env bash
# Anchor experiment round 2 — CE moved to ENCODER space (pre-projection).
#
# Round-1 finding: CE on the predictor output (proj space) fights SIGReg
# structurally — SIGReg makes that space isotropic, CE needs it class-separable.
# mlm_jepa_sigreg (0.304) < mlm_only (0.381) on MTEB mean.
#
# Fix (Austin's design): CE decodes h_masked[mask] — encoder space, which the
# architecture deliberately leaves anisotropic. JEPA MSE + SIGReg stay in proj
# space. The two objectives now meet only inside the shared encoder.
#
# Baseline for all of these: round-1 mlm_encoder_ctrl (same CE attach, no JEPA).
#
#   GPU 2:  1) enc_jepa_sigreg        — CE(enc) + MSE + SIGReg   [ours]
#           2) enc_jepa_sigreg_msew01 — same, mse_weight=0.1     [after 1]
#   GPU 3:  3) enc_jepa_ema           — CE(enc) + MSE + EMA      [Boukhari-style]
#
# Usage:  bash mask/run_anchor2.sh        (after round-1 chains finish)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/mask/env.sh"
if [ -f .env ]; then set -a; source .env; set +a; fi

mkdir -p logs

# ───────────────── shared config (matched to round 1) ───────────────────────
CORPUS="Skylion007/openwebtext"
STEPS=30000
WARMUP=1000
D_MODEL=768; D_PROJ=768; ENC_LAYERS=12; N_HEADS=12
BATCH=128; SEQLEN=128
LR=4e-4
BETA=1.0
LAM=0.001
SAVE_EVERY=0
TS="$(date +%Y%m%d_%H%M%S)"
# ─────────────────────────────────────────────────────────────────────────────

run() {
    local GPU_ID="$1"; local NAME="$2"; shift 2
    echo "════════════════════════════════════════════════════════════"
    echo "  Starting: $NAME  (GPU=$GPU_ID)"
    echo "  Args: $*"
    echo "════════════════════════════════════════════════════════════"
    CUDA_VISIBLE_DEVICES="$GPU_ID" python mask/train.py \
      --corpus "$CORPUS" --steps "$STEPS" \
      --d-model "$D_MODEL" --d-proj "$D_PROJ" \
      --enc-layers "$ENC_LAYERS" --n-heads "$N_HEADS" \
      --batch-size "$BATCH" --seq-len "$SEQLEN" \
      --lr "$LR" --warmup-steps "$WARMUP" \
      --mask-ratio 0.15 --mask-strategy random \
      --no-normalize-target \
      --mlm-beta "$BETA" --mlm-head encoder \
      --save-every "$SAVE_EVERY" \
      --wandb --mteb --run-name "$NAME" \
      "$@"
}

# GPU 2: SIGReg hybrid, then its low-MSE variant
(
  run 2 "anchor2_${TS}_enc_jepa_sigreg" \
      --mse-weight 1.0 --lam "$LAM" \
      &> "logs/anchor2_${TS}_enc_jepa_sigreg.log"
  run 2 "anchor2_${TS}_enc_jepa_sigreg_msew01" \
      --mse-weight 0.1 --lam "$LAM" \
      &> "logs/anchor2_${TS}_enc_jepa_sigreg_msew01.log"
) &
PID_GPU2=$!

# GPU 3: EMA hybrid (head-to-head vs SIGReg at the fixed attach point)
(
  run 3 "anchor2_${TS}_enc_jepa_ema" \
      --mse-weight 1.0 --lam 0.0 --ema --ema-decay 0.999 \
      &> "logs/anchor2_${TS}_enc_jepa_ema.log"
) &
PID_GPU3=$!

echo "GPU 2 chain PID: $PID_GPU2   (enc_jepa_sigreg → enc_jepa_sigreg_msew01)"
echo "GPU 3 run   PID: $PID_GPU3   (enc_jepa_ema)"
wait $PID_GPU2 && echo "==> GPU 2 chain done" || echo "==> GPU 2 chain FAILED"
wait $PID_GPU3 && echo "==> GPU 3 run done"   || echo "==> GPU 3 run FAILED"

echo ""
echo "Compare against round-1 mlm_encoder_ctrl (same CE attach, no JEPA):"
echo "  encoder_ctrl  vs  enc_jepa_sigreg  vs  enc_jepa_ema  vs  msew01"
echo "Claims: enc_jepa_sigreg > encoder_ctrl  → JEPA adds value over MLM."
echo "        enc_jepa_sigreg >= enc_jepa_ema → SIGReg replaces the EMA teacher."
