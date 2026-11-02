#!/usr/bin/env bash
# Run once per Colab session from the Colab terminal, inside the cloned repo:
#   bash scripts/colab_setup.sh asr       # Cohere + Voxtral session
#   bash scripts/colab_setup.sh moulsot   # MoulSot.0.3 session (separate deps)
#   bash scripts/colab_setup.sh           # CPU-only stages (agree, merge, labels, eval)
set -euo pipefail

EXTRA="${1:-}"
if [ -n "$EXTRA" ]; then
  pip install -q -e ".[$EXTRA,dev]"
else
  pip install -q -e ".[dev]"
fi

# Hub access: gated Cohere model + pushing run outputs to your private repo.
# Set HF_TOKEN in Colab secrets / env, or log in interactively:
if [ -z "${HF_TOKEN:-}" ]; then
  hf auth login
fi

# The LLM judge reads ANTHROPIC_API_KEY from the environment.
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "Note: ANTHROPIC_API_KEY is not set; needed only for 'darija merge'."
fi

nvidia-smi --query-gpu=name,memory.total --format=csv || echo "No GPU visible"
