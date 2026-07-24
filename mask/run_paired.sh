#!/usr/bin/env bash
# Cell 1 — cross-view paired JEPA, from scratch (EXPERIMENT_PLAN Part II).
# Does a real VIEW GAP alone rescue from-scratch latent prediction, or is the
# abstraction gap (a pretrained backbone) also required?
#
# Three arms (matched big-model config, same encoder space + SIGReg as Part I's
# winning regime). The Part I `mlm_encoder_ctrl` (0.4485) is the CE-only baseline.
#   pure     — cross-view MSE + SIGReg, NO token supervision  (view gap alone?)
#   anchored — + BERT-style CE on view A                      (LLM-JEPA in miniature)
#   shuffled — anchored but view B permuted in-batch          (chance-floor control)
#
# Reads: pure ≫ 0.164 → view gap sufficient. pure ≈ 0.164 → need abstraction gap.
#        anchored > ctrl AND shuffled ≈ ctrl → the pairing is load-bearing & faithful.
#
# Usage:  bash mask/run_paired.sh
#         PAIR=simplify bash mask/run_paired.sh          # monomodal view gap
#         GPUS="0 1 2"  bash mask/run_paired.sh          # one arm per GPU (default)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/mask/env.sh"
# Export WANDB_API_KEY from .env if present. Non-fatal: a malformed .env (or none)
# just falls through — wandb then authenticates via ~/.netrc / a prior `wandb login`.
if [ -f .env ]; then set -a; source .env 2>/dev/null || true; set +a; fi
mkdir -p logs

# ───────────────── shared config (matches run_span.sh scale) ────────────────
PAIR="${PAIR:-code}"                 # code (docstring↔code) | simplify | custom
STEPS="${STEPS:-30000}"
WARMUP=1000
D_MODEL=768; D_PROJ=768; ENC_LAYERS=12; N_HEADS=12
BATCH="${BATCH:-96}"                 # 3 encoder passes on the anchor arm — a touch
SEQLEN=128                           #   smaller than run_span's 128 to be safe
LR=4e-4
LAM=0.001
read -r -a GPU <<< "${GPUS:-0 1 2}"  # one GPU per arm
TS="$(date +%Y%m%d_%H%M%S)"
# ─────────────────────────────────────────────────────────────────────────────

run() {
    local GPU_ID="$1"; local NAME="$2"; shift 2
    echo "════════════════════════════════════════════════════════════"
    echo "  Starting: $NAME  (GPU=$GPU_ID)  pair=$PAIR"
    echo "  Args: $*"
    echo "════════════════════════════════════════════════════════════"
    CUDA_VISIBLE_DEVICES="$GPU_ID" python mask/train_paired.py \
      --pair-source "$PAIR" --steps "$STEPS" \
      --d-model "$D_MODEL" --d-proj "$D_PROJ" \
      --enc-layers "$ENC_LAYERS" --n-heads "$N_HEADS" \
      --batch-size "$BATCH" --seq-len "$SEQLEN" \
      --lr "$LR" --warmup-steps "$WARMUP" \
      --latent-space encoder --lam "$LAM" --mse-weight 1.0 \
      --wandb --mteb --run-name "$NAME" \
      "$@"
}

# One arm per GPU, in parallel (drop GPUs / add `&` chains as your box allows).
run "${GPU[0]}" "paired_${TS}_${PAIR}_pure" \
    --mlm-beta 0.0 \
    &> "logs/paired_${TS}_${PAIR}_pure.log" &
P0=$!

run "${GPU[1]:-${GPU[0]}}" "paired_${TS}_${PAIR}_anchored" \
    --mlm-beta 1.0 --mlm-head encoder \
    &> "logs/paired_${TS}_${PAIR}_anchored.log" &
P1=$!

run "${GPU[2]:-${GPU[0]}}" "paired_${TS}_${PAIR}_shuffled" \
    --mlm-beta 1.0 --mlm-head encoder --shuffle-pairs \
    &> "logs/paired_${TS}_${PAIR}_shuffled.log" &
P2=$!

echo "pure PID $P0 | anchored PID $P1 | shuffled PID $P2   (pair=$PAIR)"
wait $P0 && echo "==> pure done"     || echo "==> pure FAILED"
wait $P1 && echo "==> anchored done" || echo "==> anchored FAILED"
wait $P2 && echo "==> shuffled done" || echo "==> shuffled FAILED"

echo ""
echo "Compare MTEB means in W&B (austinmyc/lejepa) vs ctrl 0.4485 / floor 0.164."
