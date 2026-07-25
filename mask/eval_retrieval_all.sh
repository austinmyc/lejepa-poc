#!/usr/bin/env bash
# Run cross-view retrieval on every checkpoint, choosing the retrieval corpus
# from the run name (…_code_… → code, …_summary_… → summary, …_cnndm_… → cnndm;
# anything else falls back to $DEFAULT_PAIR). Single GPU, sequential — retrieval
# is cheap (a few forward passes + one N×N matmul).
#
# Usage:  bash mask/eval_retrieval_all.sh
#         CKPT_GLOB='checkpoints_mask/paired_*_final.pt' bash mask/eval_retrieval_all.sh
#         GPU=0 N_PAIRS=2000 bash mask/eval_retrieval_all.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/mask/env.sh"
if [ -f .env ]; then set -a; source .env 2>/dev/null || true; set +a; fi

CKPT_GLOB="${CKPT_GLOB:-checkpoints_mask/*_final.pt}"
GPU="${GPU:-0}"
N_PAIRS="${N_PAIRS:-1000}"
DEFAULT_PAIR="${DEFAULT_PAIR:-code}"

shopt -s nullglob
found=0
for ckpt in $CKPT_GLOB; do
    found=1
    case "$ckpt" in
        *_summary_*) pair=summary ;;
        *_cnndm_*)   pair=cnndm ;;
        *_code_*)    pair=code ;;
        *_simplify_*) pair=simplify ;;
        *)           pair="$DEFAULT_PAIR" ;;
    esac
    echo "──────────────────────────────────────────────────────────"
    CUDA_VISIBLE_DEVICES="$GPU" python mask/eval_retrieval.py \
        --ckpt "$ckpt" --pair-source "$pair" --n-pairs "$N_PAIRS" --wandb
done
[ "$found" = 1 ] || echo "No checkpoints matched: $CKPT_GLOB"
