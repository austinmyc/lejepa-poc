#!/usr/bin/env bash
# RQ4 — the JEPA-defining test: does latent prediction contribute where
# token-space supervision breaks down (long masked spans)?
#
# Fixed total mask budget (15%), varying span length L ∈ {1, 4, 8, 16}.
# Two arms per length:
#   ctrl_L — CE only (encoder attach)
#   jepa_L — CE + span-pooled latent MSE (w_span=1) + SIGReg
# Per-token MSE stays OFF in the jepa arm (redundant with CE — round-1 showed
# it neutral; the pooled term is the hypothesis). w_glob tested separately.
#
# Hypothesis: Δ(jepa_L − ctrl_L) grows with L. That curve is the headline figure.
#
#   GPU 2: all ctrl arms (L = 1, 4, 8, 16)     ~35 h
#   GPU 3: all jepa arms (L = 1, 4, 8, 16)     ~35 h
#
# Usage:  bash mask/run_span.sh
#         LENGTHS="1 8 16" bash mask/run_span.sh     # cheaper 6-run version
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
BETA=1.0
LAM=0.001
WSPAN=1.0
LENGTHS="${LENGTHS:-1 4 8 16}"
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
      --mask-ratio 0.15 --mask-strategy span \
      --no-normalize-target \
      --mlm-beta "$BETA" --mlm-head encoder \
      --wandb --mteb --run-name "$NAME" \
      "$@"
}

# GPU 2: control arms (CE only)
(
  for L in $LENGTHS; do
    run 2 "span_${TS}_L${L}_ctrl" \
        --span-len "$L" --mse-weight 0.0 --lam 0.0 \
        &> "logs/span_${TS}_L${L}_ctrl.log"
  done
) &
PID_GPU2=$!

# GPU 3: JEPA arms (CE + span-pooled latent prediction + SIGReg)
(
  for L in $LENGTHS; do
    run 3 "span_${TS}_L${L}_jepa" \
        --span-len "$L" --mse-weight 0.0 --w-span "$WSPAN" --lam "$LAM" \
        &> "logs/span_${TS}_L${L}_jepa.log"
  done
) &
PID_GPU3=$!

echo "GPU 2 chain PID: $PID_GPU2   (ctrl × {$LENGTHS})"
echo "GPU 3 chain PID: $PID_GPU3   (jepa × {$LENGTHS})"
wait $PID_GPU2 && echo "==> ctrl chain done" || echo "==> ctrl chain FAILED"
wait $PID_GPU3 && echo "==> jepa chain done" || echo "==> jepa chain FAILED"

echo ""
echo "Headline figure: plot Δ(jepa_L − ctrl_L) MTEB mean vs span length L."
echo "If Δ grows with L → latent prediction contributes where CE breaks down."
