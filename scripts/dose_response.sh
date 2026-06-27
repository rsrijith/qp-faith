#!/usr/bin/env bash
# Dose-response: 3 intermediate prompt levels x 4 planners (endpoints conservative
# & expansion already exist). 4-bit, one model at a time, ROLL-guarded. Scores v2.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONHASHSEED=0
hub="$HOME/.cache/huggingface/hub"
wait_clear(){ while pgrep -f "run_pilot.py" >/dev/null || pgrep -f "03_run_model.py" >/dev/null; do sleep 20; done; }

TAGS=(qwen4 llama4 mistral4 gemma4)
DIRS=(
  "models--mlx-community--Qwen2.5-7B-Instruct-4bit"
  "models--mlx-community--Meta-Llama-3.1-8B-Instruct-4bit"
  "models--mlx-community--Mistral-7B-Instruct-v0.3-4bit"
  "models--mlx-community--gemma-2-9b-it-4bit"
)
# mode -> short tag
MODES=(imply assist enrich)
SHORTS=(imp ast enr)

for i in "${!TAGS[@]}"; do
  m="${TAGS[$i]}"; path=$(echo "$hub/${DIRS[$i]}"/snapshots/*/)
  for j in "${!MODES[@]}"; do
    mode="${MODES[$j]}"; tag="${m}-${SHORTS[$j]}"
    [ -s "results/${tag}__scored_v2.parquet" ] && { echo "[skip] $tag"; continue; }
    wait_clear
    echo "=== RUN $tag ($mode) ==="
    python scripts/run_pilot.py --model "$path" --tag "$tag" --n 480 --mode "$mode" 2>&1 | tail -2 || { echo "FAILED $tag"; continue; }
    python scripts/score_v2.py --tag "$tag" 2>&1 | tail -3
  done
done
echo "=== DOSE-RESPONSE COMPLETE ==="
