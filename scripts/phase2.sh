#!/usr/bin/env bash
# Phase 2 driver: waits for the dose-response sweep, assembles the curve, then runs
# the ESCI secondary slice (4 models x con/exp) + scores, and recovery on the
# conservative models. ROLL-guarded on GPU steps. Run as a tracked background job.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONHASHSEED=0
DOSE_OUT="$1"
hub="$HOME/.cache/huggingface/hub"
wait_clear(){ while pgrep -f "run_pilot.py" >/dev/null || pgrep -f "run_esci.py" >/dev/null || pgrep -f "03_run_model.py" >/dev/null; do sleep 20; done; }

echo "[phase2] waiting for DOSE-RESPONSE COMPLETE..."
while ! grep -q "DOSE-RESPONSE COMPLETE" "$DOSE_OUT" 2>/dev/null; do sleep 20; done
wait_clear
echo "[phase2] assembling dose-response curve..."
python scripts/assemble.py 2>&1 | tail -12

echo "[phase2] ESCI secondary slice..."
TAGS=(qwen4 llama4 mistral4 gemma4)
DIRS=(
  "models--mlx-community--Qwen2.5-7B-Instruct-4bit"
  "models--mlx-community--Meta-Llama-3.1-8B-Instruct-4bit"
  "models--mlx-community--Mistral-7B-Instruct-v0.3-4bit"
  "models--mlx-community--gemma-2-9b-it-4bit"
)
for i in "${!TAGS[@]}"; do
  m="${TAGS[$i]}"; path=$(echo "$hub/${DIRS[$i]}"/snapshots/*/)
  for mode in conservative expansion; do
    short=$([ "$mode" = conservative ] && echo con || echo exp)
    tag="${m}-${short}"
    [ -s "results/esci-${tag}__scored.parquet" ] && { echo "[skip] esci-$tag"; continue; }
    wait_clear
    echo "=== ESCI $tag ($mode) ==="
    python scripts/run_esci.py --model "$path" --tag "$tag" --mode "$mode" 2>&1 | tail -2 || { echo "FAILED esci-$tag"; continue; }
    python scripts/score_esci.py --tag "$tag" 2>&1 | tail -8
  done
done

echo "[phase2] recovery on conservative models (CPU)..."
for t in qwen4-con llama4-con mistral4-con gemma4-con; do
  [ -s "results/${t}__recovery.json" ] && continue
  python scripts/score_recovery.py --tag "$t" 2>&1 | tail -3
done
echo "=== PHASE2 COMPLETE ==="
