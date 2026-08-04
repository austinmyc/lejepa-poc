#!/usr/bin/env bash
# LLM-JEPA from-scratch NL-RX-SYNTH: reproduction + the control they never ran.
#
#   B0_base   λ=0                      NTP only            (paper: 54.38 ± 1.70)
#   B1_jepa   λ=1                      LLM-JEPA            (paper: 60.59 ± 1.01)
#   B2_shuf   λ=1 + --shuffle-pairs    each description paired with a RANDOM
#                                      other regex in the JEPA term only
#
# The read: if B2 ≈ B1, the published gain is not coming from the semantic
# pairing — it is the auxiliary loss acting as a regulariser. If B2 ≈ B0, the
# pairing is genuinely load-bearing and the paper's attribution is correct.
#
# NOTE ON SCALE: "pretraining from random init" here is ~6.5k examples ≈ 7M
# tokens for 30 epochs — five orders of magnitude below real LLM pretraining.
# One 1B arm fits one 44GB GPU (~25GB) and runs in a few hours.
#
# Usage:  GPUS="0 2 3" bash ft/run_nlrx.sh
#         SEEDS="1337 2024 7" GPUS="0 2 3" bash ft/run_nlrx.sh   # error bars
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -f mask/env.sh ] && source mask/env.sh || true
if [ -f .env ]; then set -a; source .env 2>/dev/null || true; set +a; fi
mkdir -p logs

MODEL="${MODEL:-unsloth/Llama-3.2-1B}"     # architecture only — weights are RANDOM
EPOCHS="${EPOCHS:-4}"        # their code: "we fix number of epochs to 4"
BATCH="${BATCH:-32}"
KPRED="${KPRED:-3}"          # paper's pretraining k=3
LR="${LR:-8e-5}"            # paper's pretraining lr
LAM="${LAM:-2.0}"           # paper's pretraining lambda=2
SEEDS="${SEEDS:-1337}"
read -r -a GPU <<< "${GPUS:-0 2 3}"
NGPU=${#GPU[@]}
TS="$(date +%Y%m%d_%H%M%S)"

run() {  # run <gpu> <name> <extra flags...>
    local g="$1"; local name="$2"; shift 2
    echo "═══ $name (GPU=$g) ═══"
    CUDA_VISIBLE_DEVICES="$g" python ft/train_llmjepa.py \
        --model "$MODEL" --epochs "$EPOCHS" --batch-size "$BATCH" --lr "$LR" \
        --n-pred-tokens "$KPRED" \
        --grad-ckpt --run-name "$name" --wandb "$@" \
    && CUDA_VISIBLE_DEVICES="$g" python ft/eval_nlrx.py \
        --ckpt "ft_checkpoints/$name" --split test --dfa --wandb --run-name "eval_$name"
}

i=0
for SEED in $SEEDS; do
  for ARM in B0_base B1_jepa B2_shuf; do
      case "$ARM" in
        B0_base) FLAGS="--lam 0" ;;
        B1_jepa) FLAGS="--lam $LAM" ;;
        B2_shuf) FLAGS="--lam $LAM --shuffle-pairs" ;;
      esac
      NAME="nlrx_${TS}_${ARM}_s${SEED}"
      # shellcheck disable=SC2086
      run "${GPU[$((i % NGPU))]}" "$NAME" --seed "$SEED" $FLAGS \
          &> "logs/${NAME}.log" &
      echo "  $NAME → GPU ${GPU[$((i % NGPU))]} (PID $!)"
      i=$((i + 1))
      if (( i % NGPU == 0 )); then wait; fi
  done
done
wait
echo "done — compare B0 / B1 / B2 test accuracy in W&B (austinmyc/lejepa-ft)"
