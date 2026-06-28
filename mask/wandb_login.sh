#!/usr/bin/env bash
# Connect to Weights & Biases. Reads WANDB_API_KEY from the environment or from
# a .env file at the repo root (the .env is gitignored, so it won't be in a
# fresh clone — create it on the server, or export the key first).
# Usage:  bash mask/wandb_login.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

# Load .env if present (KEY=VALUE lines).
if [ -f .env ]; then
  set -a; source .env; set +a
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
  echo "==> Logging in with WANDB_API_KEY"
  wandb login "$WANDB_API_KEY"
else
  echo "WANDB_API_KEY not found."
  echo "Either:  echo 'WANDB_API_KEY=<your-key>' > .env"
  echo "    or:  export WANDB_API_KEY=<your-key>"
  echo "    or:  run 'wandb login' interactively."
  exit 1
fi

echo "==> W&B ready. Runs will sync to project 'lejepa' (entity in config.py)."
