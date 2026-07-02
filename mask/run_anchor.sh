#!/usr/bin/env bash
# MLM-anchor experiment — the matched-compute comparison that tests the claim:
#   "MLM + JEPA latent prediction + SIGReg beats MLM alone, without an EMA teacher."
#
# Four runs, same architecture and compute:
#   GPU 2:  1) mlm_only        — CE on predictor path only (matched-path baseline)
#           2) mlm_jepa        — CE + JEPA MSE (no SIGReg)          [after 1]
#   GPU 3:  3) mlm_jepa_sigreg — CE + JEPA MSE + SIGReg (full model)
#           4) mlm_encoder_ctrl — CE directly on encoder output     [after 3]
#              (standard BERT-style MLM: the honest external control — the
#               encoder-readout number any anchored-JEPA claim must beat)
#
# Usage:  bash mask/run_anchor.sh
#         nohup bash mask/run_anchor.sh &> anchor.log &
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/mask/env.sh"
if [ -f .env ]; then set -a; source .env; set +a; fi

mkdir -p logs

# ───────────────── shared config ────────────────────────────────────────────
CORPUS="Skylion007/openwebtext"
STEPS=30000
WARMUP=1000
D_MODEL=768; D_PROJ=768; ENC_LAYERS=12; N_HEADS=12
BATCH=128; SEQLEN=128
LR=4e-4
BETA=1.0        # MLM anchor weight (CE ~10 nats at init; MSE ~1 — CE dominates early, which is the point)
LAM=0.001       # only stable lam (sweep C — best MTEB so far)
MASK_RATIO=0.15 # keep BERT-style for comparability with sweep C; ratio is a separate axis
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
      --mask-ratio "$MASK_RATIO" --mask-strategy random \
      --no-normalize-target \
      --save-every "$SAVE_EVERY" \
      --wandb --mteb --run-name "$NAME" \
      "$@"
}

# GPU 2: pure MLM baseline, then MLM+JEPA (sequential)
(
  run 2 "anchor_${TS}_mlm_only" \
      --mlm-beta "$BETA" --mse-weight 0.0 --lam 0.0 \
      &> "logs/anchor_${TS}_mlm_only.log"
  run 2 "anchor_${TS}_mlm_jepa" \
      --mlm-beta "$BETA" --mse-weight 1.0 --lam 0.0 \
      &> "logs/anchor_${TS}_mlm_jepa.log"
) &
PID_GPU2=$!

# GPU 3: full model, then the BERT-style encoder-CE control (sequential)
(
  run 3 "anchor_${TS}_mlm_jepa_sigreg" \
      --mlm-beta "$BETA" --mse-weight 1.0 --lam "$LAM" \
      &> "logs/anchor_${TS}_mlm_jepa_sigreg.log"
  run 3 "anchor_${TS}_mlm_encoder_ctrl" \
      --mlm-beta "$BETA" --mse-weight 0.0 --lam 0.0 --mlm-head encoder \
      &> "logs/anchor_${TS}_mlm_encoder_ctrl.log"
) &
PID_GPU3=$!

echo "GPU 2 chain PID: $PID_GPU2   (mlm_only → mlm_jepa)"
echo "GPU 3 chain PID: $PID_GPU3   (mlm_jepa_sigreg → mlm_encoder_ctrl)"
wait $PID_GPU2 && echo "==> GPU 2 chain done" || echo "==> GPU 2 chain FAILED"
wait $PID_GPU3 && echo "==> GPU 3 chain done" || echo "==> GPU 3 chain FAILED"

echo ""
echo "The comparison that matters (MTEB mean, W&B 'anchor_${TS}_*'):"
echo "  mlm_encoder_ctrl (BERT-style MLM)  vs  mlm_only  vs  mlm_jepa  vs  mlm_jepa_sigreg"
echo "Headline claim needs: mlm_jepa_sigreg > mlm_encoder_ctrl (encoder readout)."
