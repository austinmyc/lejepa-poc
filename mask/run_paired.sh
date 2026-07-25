#!/usr/bin/env bash
# Cell 1 — cross-view paired JEPA, from scratch (EXPERIMENT_PLAN Part II).
# Does a real VIEW GAP alone rescue from-scratch latent prediction, or is the
# abstraction gap (a pretrained backbone) also required?
#
# Arms (matched big-model config; the Part I `mlm_encoder_ctrl` 0.4485 @ batch128
# is the single-text CE baseline, floor 0.164):
#   pure     — cross-view MSE + SIGReg, NO token supervision   (view gap alone?)
#   anchored — + BERT-style CE on view A                       (LeJEPA cross-view)
#   shuffled — anchored but view B permuted in-batch           (pairing control)
#   llmjepa  — anchored WITHOUT SIGReg (lam=0)                 (faithful LLM-JEPA
#              from scratch: NTP + predictor-MSE, no isotropy term — rules out
#              "our SIGReg sterilised it" as the reason for a null)
#   mlmonly  — CE only in the paired pipeline (mse-weight 0, lam 0, batch matched)
#              (batch-96 baseline so "cross-view hurts" is apples-to-apples)
#
# Reads: anchored≈shuffled → pairing not load-bearing. llmjepa≈shuffled → the null
#        survives even without SIGReg (it's from-scratch, not the regulariser).
#        anchored/llmjepa vs mlmonly → does cross-view help/hurt at matched batch?
#
# Usage:  bash mask/run_paired.sh                              # all 5 arms, code+summary
#         ARMS="llmjepa mlmonly" bash mask/run_paired.sh       # just the new arms
#         PAIRS="code summary" GPUS="0 2 3" bash mask/run_paired.sh
#
# Arms are scheduled in batches of |GPUS| per pair source (one arm per GPU at a
# time); pair sources run as sequential waves.
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
PAIRS="${PAIRS:-code summary}"       # pair sources to sweep (space-separated)
ARMS="${ARMS:-pure anchored shuffled llmjepa mlmonly}"   # arms to run
STEPS="${STEPS:-30000}"
WARMUP=1000
D_MODEL=768; D_PROJ=768; ENC_LAYERS=12; N_HEADS=12
BATCH="${BATCH:-96}"                 # kept fixed across arms so comparisons are
SEQLEN=128                           #   batch-matched (the mlmonly baseline too)
LR=4e-4
LAM="${LAM:-0.001}"                  # SIGReg weight for SIGReg-bearing arms
read -r -a GPU <<< "${GPUS:-0 1 2}"
NGPU=${#GPU[@]}
TS="$(date +%Y%m%d_%H%M%S)"
# ─────────────────────────────────────────────────────────────────────────────

# Per-arm objective flags (shared invariants live in run()).
arm_flags() {
    case "$1" in
        pure)     echo "--mlm-beta 0.0 --lam $LAM" ;;
        anchored) echo "--mlm-beta 1.0 --mlm-head encoder --lam $LAM" ;;
        shuffled) echo "--mlm-beta 1.0 --mlm-head encoder --lam $LAM --shuffle-pairs" ;;
        llmjepa)  echo "--mlm-beta 1.0 --mlm-head encoder --lam 0.0" ;;
        mlmonly)  echo "--mlm-beta 1.0 --mlm-head encoder --lam 0.0 --mse-weight 0.0" ;;
        *) echo "__UNKNOWN_ARM__" ;;
    esac
}

run() {
    local GPU_ID="$1"; local PAIR="$2"; local NAME="$3"; shift 3
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
      --latent-space encoder \
      --wandb --mteb --run-name "$NAME" \
      "$@"
}

# Wait for the currently-launched batch, reporting each arm's exit status. Reads
# the caller's `pids`/`arms` via bash dynamic scope (called only from run_wave).
_flush() {
    local j
    for j in "${!pids[@]}"; do
        wait "${pids[$j]}" && echo "==> [${arms[$j]}] done" || echo "==> [${arms[$j]}] FAILED"
    done
    pids=(); arms=()
}

# Launch the requested arms for one pair source, |GPUS| at a time.
run_wave() {
    local PAIR="$1"; local i=0
    local pids=(); local arms=()
    for ARM in $ARMS; do
        local flags; flags="$(arm_flags "$ARM")"
        if [ "$flags" = "__UNKNOWN_ARM__" ]; then
            echo "!! skipping unknown arm: $ARM" >&2; continue
        fi
        local gpu="${GPU[$((i % NGPU))]}"
        local name="paired_${TS}_${PAIR}_${ARM}"
        # shellcheck disable=SC2086
        run "$gpu" "$PAIR" "$name" $flags &> "logs/${name}.log" &
        pids+=("$!"); arms+=("$ARM")
        i=$((i + 1))
        if (( i % NGPU == 0 )); then _flush; fi
    done
    _flush
}

for PAIR in $PAIRS; do
    echo "########## PAIR SOURCE: $PAIR  (arms: $ARMS) ##########"
    run_wave "$PAIR"
done

echo ""
echo "Compare in W&B (austinmyc/lejepa): MTEB means vs ctrl 0.4485 / floor 0.164."
echo "Then run eval_retrieval.py on the checkpoints for the cross-view read"
echo "(does anchored beat shuffled on code↔docstring / doc↔summary retrieval?)."
