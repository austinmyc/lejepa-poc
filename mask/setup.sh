#!/usr/bin/env bash
# Server setup: create a venv and install dependencies.
# Usage:  bash mask/setup.sh
set -euo pipefail

# Resolve repo root (this script lives in mask/).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Creating virtualenv at $ROOT/.venv"
python3 -m venv .venv
source .venv/bin/activate

echo "==> Upgrading pip and installing requirements"
pip install --upgrade pip
# NOTE: on a CUDA server the default torch wheel includes CUDA. If you need a
# specific CUDA build, install torch first from https://pytorch.org, then:
pip install -r requirements.txt

echo "==> Done. Next:"
echo "    source .venv/bin/activate"
echo "    bash mask/wandb_login.sh      # connect Weights & Biases"
echo "    bash mask/run_owt.sh          # launch the OpenWebText run"
