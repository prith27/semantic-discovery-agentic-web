#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p figures results

# Offline sweep (fast, no API)
python -m src.run_all --seed 0 --n-agents 300 --n-tasks 80 --skill-dim 12

# Paper sweep (requires ANTHROPIC_API_KEY in .env)
if [[ -f .env ]] && grep -q 'ANTHROPIC_API_KEY=sk-' .env 2>/dev/null; then
  echo "Running PAPER mode with Anthropic API..."
  python -m src.run_all --paper --n-agents 200 --api-tasks 24 --ks 3 5 --claude-spam-max 25
else
  echo "Skip paper/API run: add ANTHROPIC_API_KEY to .env then re-run:"
  echo "  python -m src.run_all --paper"
fi

echo "Done. See figures/ and results/"
