#!/usr/bin/env bash
# Sweep the SIGReg→encoder gradient gate α on OpenWebText.
# Goal: find where the projection absorbs the isotropy shaping while the encoder
# stays anisotropic — compare enc_iso vs proj_iso (and iso_gap) across runs in W&B.
# Shorter step budget than run_owt.sh — this is exploration, not the final run.
# Usage:  GPU=0 bash mask/sweep_alpha.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
if [ -f .env ]; then set -a; source .env; set +a; fi

# ───────────────── sweep config — edit these ────────────────────────────────
GPU="${GPU:-0}"
ALPHAS=(1.0 0.3 0.0)              # gate values: full → partial → fully shielded
STEPS=10000                       # exploration budget (final run uses run_owt.sh)
CORPUS="Skylion007/openwebtext"
D_MODEL=768; D_PROJ=768; ENC_LAYERS=12; N_HEADS=12
BATCH=32; SEQLEN=128; LAM=0.006; SAVE_EVERY=5000
# ─────────────────────────────────────────────────────────────────────────────

export CUDA_VISIBLE_DEVICES="$GPU"
echo "==> α sweep on GPU=$GPU: ${ALPHAS[*]}  ($STEPS steps each, $CORPUS)"

for A in "${ALPHAS[@]}"; do
  RUN="lejepa_mask_alpha${A}_$(date +%Y%m%d_%H%M%S)"
  echo "==> α=$A  run=$RUN"
  python mask/train.py \
    --corpus "$CORPUS" --steps "$STEPS" \
    --d-model "$D_MODEL" --d-proj "$D_PROJ" \
    --enc-layers "$ENC_LAYERS" --n-heads "$N_HEADS" \
    --batch-size "$BATCH" --seq-len "$SEQLEN" \
    --lam "$LAM" --sigreg-grad-scale "$A" \
    --save-every "$SAVE_EVERY" \
    --wandb --run-name "$RUN"
done

echo "==> Sweep done. In W&B, compare enc_iso / proj_iso / iso_gap across the runs:"
echo "    iso_gap > 0 and rising = projection absorbs isotropy, encoder stays anisotropic."
