#!/usr/bin/env bash
# Part III — DLLM-JEPA reproduction, then CoT as the predictor's action.
#
# TWO STAGES, and they use different generation targets — do not mix them:
#
#   REPRODUCTION (--cot-in-target: standard GSM8K SFT, model generates the CoT)
#     R0 sft   λ=0                     baseline
#     R1 jepa  λ=1, action=none        DLLM-JEPA  ← GATE 1: R1 must beat R0
#                                      (paper: LLaDA-8B 42.61 → 61.33, +18.7pp)
#
#   ACTION EXPERIMENT (answer-only target ⇒ no CoT generated at test time;
#   the CoT enters ONLY as the predictor's action — the paper's contribution)
#     A1 jepa       λ=1, action=none       no-action reference
#     A2 jepa_cot   λ=1, action=cot        THE CONTRIBUTION
#     A3 jepa_shuf  λ=1, action=shuffled   faithfulness control
#     A4 jepa_rand  λ=1, action=random     any-conditioning control
#     A5 cot_sft    λ=0, --cot-in-target   = R0, the honest comparator
#
# Reading: A2 > A3 ≈ A4 ≈ A1 → CoT carries example-specific transport signal.
#          A2 ≈ A5 accuracy with NO test-time CoT → the efficiency claim.
#
# Usage:  STAGE=repro bash ft/run.sh          # gate first, always
#         STAGE=action bash ft/run.sh
#         MODEL=... GPUS="0 2 3" bash ft/run.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -f mask/env.sh ] && source mask/env.sh || true
if [ -f .env ]; then set -a; source .env 2>/dev/null || true; set +a; fi
mkdir -p logs

MODEL="${MODEL:-GSAI-ML/LLaDA-8B-Instruct}"
STAGE="${STAGE:-repro}"
LAM="${LAM:-1.0}"
EPOCHS="${EPOCHS:-2}"
BATCH="${BATCH:-2}"
ACCUM="${ACCUM:-8}"
LR="${LR:-1e-5}"
MASK_ID="${MASK_ID:-126336}"          # LLaDA-8B; 0 = auto-detect
EVAL_N="${EVAL_N:-200}"
read -r -a GPU <<< "${GPUS:-0 2 3}"
NGPU=${#GPU[@]}
TS="$(date +%Y%m%d_%H%M%S)"

run() {   # run <gpu> <name> <extra flags...>
    local g="$1"; local name="$2"; shift 2
    echo "═══ $name (GPU=$g) ═══"
    CUDA_VISIBLE_DEVICES="$g" python ft/train.py \
        --model "$MODEL" --mask-token-id "$MASK_ID" \
        --epochs "$EPOCHS" --batch-size "$BATCH" --grad-accum "$ACCUM" --lr "$LR" \
        --run-name "$name" --wandb "$@" \
    && CUDA_VISIBLE_DEVICES="$g" python ft/eval_gsm8k.py \
        --model "$MODEL" --adapter "ft_checkpoints/$name" \
        --mask-token-id "$MASK_ID" --eval-n "$EVAL_N" --wandb --run-name "eval_$name"
}

launch() { # launch <idx> <name> <flags...> — round-robins GPUs, backgrounds
    local i="$1"; local name="$2"; shift 2
    run "${GPU[$((i % NGPU))]}" "$name" "$@" &> "logs/ft_${name}.log" &
    echo "  $name → GPU ${GPU[$((i % NGPU))]}  (PID $!)"
}

if [ "$STAGE" = "repro" ]; then
    echo "### GATE 1 — reproduce the published DLLM-JEPA gain ###"
    launch 0 "ft_${TS}_R0_sft"  --lam 0 --cot-in-target
    launch 1 "ft_${TS}_R1_jepa" --lam "$LAM" --action none --cot-in-target
else
    echo "### ACTION EXPERIMENT — CoT as the predictor's action ###"
    launch 0 "ft_${TS}_A1_jepa"      --lam "$LAM" --action none
    launch 1 "ft_${TS}_A2_jepa_cot"  --lam "$LAM" --action cot
    launch 2 "ft_${TS}_A3_jepa_shuf" --lam "$LAM" --action shuffled
    launch 3 "ft_${TS}_A4_jepa_rand" --lam "$LAM" --action random
fi
wait
echo "done — compare GSM8K accuracy in W&B (${STAGE}); logs in logs/ft_*.log"
