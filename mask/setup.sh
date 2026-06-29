#!/usr/bin/env bash
# Server setup: create a conda env and install dependencies.
# Usage:  bash mask/setup.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_NAME="${CONDA_ENV_NAME:-lejepa-poc}"

# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "==> Conda env '$ENV_NAME' already exists"
else
  echo "==> Creating conda env '$ENV_NAME' (python 3.12)"
  conda create -n "$ENV_NAME" python=3.12 -y
fi

conda activate "$ENV_NAME"

echo "==> Upgrading pip and installing requirements"
pip install --upgrade pip
# NOTE: on a CUDA server the default torch wheel includes CUDA. If you need a
# specific CUDA build, install torch first from https://pytorch.org, then:
pip install -r requirements.txt

echo "==> Done. Next:"
echo "    conda activate $ENV_NAME"
echo "    bash mask/wandb_login.sh      # connect Weights & Biases"
echo "    bash mask/run_owt.sh          # launch the OpenWebText run"
