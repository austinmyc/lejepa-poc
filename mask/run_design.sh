#!/usr/bin/env bash
# Wave 1 — validate the designed loss at one span length (L=8), with
# credit attribution built in. Four arms, one per GPU, ~9 h total.
#
#   L = β·CE(dec(h_masked[M]), y_M)      anchor, encoder space
#     + w_s·L_span + w_g·L_glob          JEPA, proj space (spans / global)
#     + λ·SIGReg(z_clean)                geometry, proj space
#
#   GPU 0: ctrl       — CE only                       (baseline @ L=8 masking)
#   GPU 1: jepa_full  — CE + span + glob + SIGReg     (the designed loss)
#   GPU 2: jepa_span  — CE + span + SIGReg            (composition term alone)
#   GPU 3: jepa_glob  — CE + glob + SIGReg            (readout term alone)
#
# Reading: full > ctrl → design works; span/glob arms attribute the credit.
# ctrl & jepa_span double as the L=8 pair of the RQ4 sweep (run_span.sh).
#
# Usage:  bash mask/run_design.sh
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
WGLOB=0.5
SPANLEN=8
TS="$(date +%Y%m%d_%H%M%S)"
# ─────────────────────────────────────────────────────────────────────────────

run() {
    local GPU_ID="$1"; local NAME="$2"; shift 2
    CUDA_VISIBLE_DEVICES="$GPU_ID" nohup python mask/train.py \
      --corpus "$CORPUS" --steps "$STEPS" \
      --d-model "$D_MODEL" --d-proj "$D_PROJ" \
      --enc-layers "$ENC_LAYERS" --n-heads "$N_HEADS" \
      --batch-size "$BATCH" --seq-len "$SEQLEN" \
      --lr "$LR" --warmup-steps "$WARMUP" \
      --mask-ratio 0.15 --mask-strategy span --span-len "$SPANLEN" \
      --no-normalize-target \
      --mlm-beta "$BETA" --mlm-head encoder --mse-weight 0.0 \
      --save-every 10000 \
      --wandb --mteb --run-name "$NAME" \
      "$@" &> "logs/${NAME}.log" &
    echo "  launched $NAME on GPU $GPU_ID (PID $!)"
}

run 0 "design_${TS}_L8_ctrl"                                    --lam 0.0
run 1 "design_${TS}_L8_jepa_full"  --w-span "$WSPAN" --w-glob "$WGLOB" --lam "$LAM"
run 2 "design_${TS}_L8_jepa_span"  --w-span "$WSPAN"                   --lam "$LAM"
run 3 "design_${TS}_L8_jepa_glob"                   --w-glob "$WGLOB"  --lam "$LAM"

echo ""
echo "All four launched. Verdict table when MTEB lands:"
echo "  ctrl  vs  jepa_span  vs  jepa_glob  vs  jepa_full"
wait
echo "==> wave 1 complete"
