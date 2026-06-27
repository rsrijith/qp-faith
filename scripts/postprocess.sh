#!/usr/bin/env bash
# Waits for the cross-model sweep to finish, then re-scores all models with the
# validity-fixed pipeline (v2 + segment + soft) and runs the procedural probe
# through all 4 planners. Guarded against ROLL (03_run_model.py) for the GPU steps.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONHASHSEED=0
SWEEP_OUT="$1"   # path to the sweep's output file

wait_clear() {  # wait until no run_pilot.py and no ROLL 03_run_model.py
  while pgrep -f "run_pilot.py" >/dev/null || pgrep -f "03_run_model.py" >/dev/null; do sleep 20; done
}

echo "[postprocess] waiting for sweep COMPLETE marker..."
while ! grep -q "CROSS-MODEL SWEEP COMPLETE" "$SWEEP_OUT" 2>/dev/null; do sleep 20; done
wait_clear
echo "[postprocess] sweep done. re-scoring all tags with v2/seg/soft..."

TAGS="qwen4-con qwen4-exp llama4-con llama4-exp mistral4-con mistral4-exp gemma4-con gemma4-exp"
for t in $TAGS; do
  [ -s "results/${t}__plans.jsonl" ] || { echo "[skip] no plans for $t"; continue; }
  echo "=== rescore $t ==="
  python scripts/score_v2.py   --tag "$t" 2>&1 | tail -8
  python scripts/seg_report.py --tag "$t" 2>&1 | head -5
  python scripts/score_soft.py --tag "$t" 2>&1 | tail -4
done

echo "[postprocess] running procedural probe through 4 planners (expansion)..."
hub="$HOME/.cache/huggingface/hub"
TAGS2=(qwen4 llama4 mistral4 gemma4)
DIRS=(
  "models--mlx-community--Qwen2.5-7B-Instruct-4bit"
  "models--mlx-community--Meta-Llama-3.1-8B-Instruct-4bit"
  "models--mlx-community--Mistral-7B-Instruct-v0.3-4bit"
  "models--mlx-community--gemma-2-9b-it-4bit"
)
for i in "${!TAGS2[@]}"; do
  m="${TAGS2[$i]}"; path=$(echo "$hub/${DIRS[$i]}"/snapshots/*/)
  for mode in conservative expansion; do
    short=$([ "$mode" = conservative ] && echo con || echo exp)
    tag="probe-${m}-${short}"
    [ -s "results/${tag}__probe_scored.parquet" ] && { echo "[skip] $tag"; continue; }
    wait_clear
    echo "=== PROBE $tag ($mode) ==="
    python scripts/run_probe.py --model "$path" --tag "$tag" --mode "$mode" 2>&1 | tail -3
    python scripts/score_probe.py --tag "$tag" 2>&1 | tail -6
  done
done
echo "=== POSTPROCESS COMPLETE ==="
