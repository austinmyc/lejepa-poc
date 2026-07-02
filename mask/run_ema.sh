#!/usr/bin/env bash
# Launch two EMA runs in parallel:
#   GPU 2 — EMA + SIGReg (lam=0.001)
#   GPU 3 — EMA only     (lam=0, pure MSE target from teacher)
#
# Usage:  bash mask/run_ema.sh
#         nohup bash mask/run_ema.sh &> ema.log &
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/mask/env.sh"
if [ -f .env ]; then set -a; source .env; set +a; fi

# ───────────────── shared config ────────────────────────────────────────────
CORPUS="Skylion007/openwebtext"
STEPS=30000
WARMUP=1000
D_MODEL=768; D_PROJ=768; ENC_LAYERS=12; N_HEADS=12
BATCH=128; SEQLEN=128
LR=4e-4
EMA_DECAY=0.999
SAVE_EVERY=0
TS="$(date +%Y%m%d_%H%M%S)"
# ─────────────────────────────────────────────────────────────────────────────

NAME_SIGREG="ema_sigreg_${TS}"
NAME_PURE="ema_pure_${TS}"

echo "==> Launching EMA+SIGReg on GPU 2: $NAME_SIGREG"
CUDA_VISIBLE_DEVICES=2 python mask/train.py \
  --corpus "$CORPUS" --steps "$STEPS" \
  --d-model "$D_MODEL" --d-proj "$D_PROJ" \
  --enc-layers "$ENC_LAYERS" --n-heads "$N_HEADS" \
  --batch-size "$BATCH" --seq-len "$SEQLEN" \
  --lr "$LR" --warmup-steps "$WARMUP" \
  --lam 0.001 \
  --ema --ema-decay "$EMA_DECAY" \
  --save-every "$SAVE_EVERY" \
  --wandb --mteb --run-name "$NAME_SIGREG" \
  &> "logs/${NAME_SIGREG}.log" &

PID_SIGREG=$!
echo "   PID: $PID_SIGREG  |  log: logs/${NAME_SIGREG}.log"

echo "==> Launching EMA-only on GPU 3:    $NAME_PURE"
CUDA_VISIBLE_DEVICES=3 python mask/train.py \
  --corpus "$CORPUS" --steps "$STEPS" \
  --d-model "$D_MODEL" --d-proj "$D_PROJ" \
  --enc-layers "$ENC_LAYERS" --n-heads "$N_HEADS" \
  --batch-size "$BATCH" --seq-len "$SEQLEN" \
  --lr "$LR" --warmup-steps "$WARMUP" \
  --lam 0.0 \
  --ema --ema-decay "$EMA_DECAY" \
  --save-every "$SAVE_EVERY" \
  --wandb --mteb --run-name "$NAME_PURE" \
  &> "logs/${NAME_PURE}.log" &

PID_PURE=$!
echo "   PID: $PID_PURE  |  log: logs/${NAME_PURE}.log"

echo ""
echo "Both runs launched. Waiting for completion..."
wait $PID_SIGREG && echo "==> EMA+SIGReg done: $NAME_SIGREG" || echo "==> EMA+SIGReg FAILED (PID $PID_SIGREG)"
wait $PID_PURE   && echo "==> EMA-only done:   $NAME_PURE"   || echo "==> EMA-only FAILED (PID $PID_PURE)"

echo ""
echo "MTEB results written to mteb_results/ and logged to W&B."
