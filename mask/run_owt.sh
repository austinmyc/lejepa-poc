#!/usr/bin/env bash
# Launch the full OpenWebText training run.
# Usage:  bash mask/run_owt.sh
# Long-running — consider `tmux`/`screen`, or:  nohup bash mask/run_owt.sh &> owt.log &
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
if [ -f .env ]; then set -a; source .env; set +a; fi   # for WANDB_API_KEY

# ───────────────── run config — edit these ──────────────────────────────────
GPU="${GPU:-0}"                   # which GPU to use; run `nvidia-smi` to list indices
CORPUS="Skylion007/openwebtext"   # namespaced HF repo; or "HuggingFaceFW/fineweb"
STEPS=50000                       # ~500M tokens at batch 32 × seq 128
# Model scale: 768/12L ≈ BERT-base (~110M), matches embedding-model peers.
# Drop to 256/4L for a faster, cheaper first run.
D_MODEL=768
D_PROJ=768                        # >= D_MODEL (expander), per methods_note — not a bottleneck
ENC_LAYERS=12
N_HEADS=12
BATCH=32
SEQLEN=128
LAM=0.006                         # SIGReg weight (calibrated ~3x encoder grad ratio)
ALPHA=1.0                         # SIGReg gradient into encoder: 1=full, <1 shields it, 0=none
SAVE_EVERY=5000
RUN_NAME="lejepa_mask_owt_$(date +%Y%m%d_%H%M%S)"
# ─────────────────────────────────────────────────────────────────────────────

export CUDA_VISIBLE_DEVICES="$GPU"   # pin to a single GPU (the L20)
echo "==> Launching: $RUN_NAME  (GPU=$GPU, corpus=$CORPUS, steps=$STEPS, d_model=$D_MODEL)"
python mask/train.py \
  --corpus "$CORPUS" \
  --steps "$STEPS" \
  --d-model "$D_MODEL" --d-proj "$D_PROJ" \
  --enc-layers "$ENC_LAYERS" --n-heads "$N_HEADS" \
  --batch-size "$BATCH" --seq-len "$SEQLEN" \
  --lam "$LAM" \
  --sigreg-grad-scale "$ALPHA" \
  --save-every "$SAVE_EVERY" \
  --wandb --run-name "$RUN_NAME"

echo "==> Done. Evaluate the final checkpoint:"
echo "    python mask/eval_sts.py checkpoints_mask/${RUN_NAME}_final.pt"
