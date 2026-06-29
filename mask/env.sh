#!/usr/bin/env bash
# Activate the project conda env. Sourced by setup/wandb_login/run scripts.
# Override with:  CONDA_ENV_NAME=myenv bash mask/setup.sh

ENV_NAME="${CONDA_ENV_NAME:-lejepa-poc}"

if [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "$CONDA_DEFAULT_ENV" = "$ENV_NAME" ]; then
  return 0 2>/dev/null || exit 0
fi

if [ -f "${CONDA_EXE:-}/../etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$(dirname "$CONDA_EXE")/../etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "Could not find conda. Install miniconda or add it to PATH." >&2
  return 1 2>/dev/null || exit 1
fi

conda activate "$ENV_NAME"
