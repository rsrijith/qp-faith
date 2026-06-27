#!/usr/bin/env bash
# Cross-model sweep: 4 planners x 2 regimes, 4-bit, ONE model in memory at a time
# (24GB unified-memory ceiling). Scores + segments after each run.
# bash 3.2 compatible (macOS default): no associative arrays.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONHASHSEED=0

hub="$HOME/.cache/huggingface/hub"
# parallel arrays: short tag + hub dir name
TAGS=(qwen4 llama4 mistral4 gemma4)
DIRS=(
  "models--mlx-community--Qwen2.5-7B-Instruct-4bit"
  "models--mlx-community--Meta-Llama-3.1-8B-Instruct-4bit"
  "models--mlx-community--Mistral-7B-Instruct-v0.3-4bit"
  "models--mlx-community--gemma-2-9b-it-4bit"
)

for i in "${!TAGS[@]}"; do
  m="${TAGS[$i]}"
  path=$(echo "$hub/${DIRS[$i]}"/snapshots/*/)
  for mode in conservative expansion; do
    if [ "$mode" = conservative ]; then short=con; else short=exp; fi
    tag="${m}-${short}"
    if [ -s "results/${tag}__scored.parquet" ]; then
      echo "[skip] $tag already scored"; continue
    fi
    echo "=== RUN $tag ($mode) $path ==="
    python scripts/run_pilot.py --model "$path" --tag "$tag" --n 480 --mode "$mode" || { echo "FAILED $tag"; continue; }
    python scripts/score.py --tag "$tag"
    python scripts/seg_report.py --tag "$tag"
  done
done
echo "=== CROSS-MODEL SWEEP COMPLETE ==="
