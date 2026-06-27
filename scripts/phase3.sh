#!/usr/bin/env bash
# Phase 3: second-domain ESCI entity probe + schema-nullability ablation, on the 4
# local planners. Waits for phase2, ROLL-guarded. Tracked background job.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONHASHSEED=0
PHASE2_OUT="$1"
hub="$HOME/.cache/huggingface/hub"
wait_clear(){ while pgrep -f "run_pilot.py" >/dev/null || pgrep -f "run_probe.py" >/dev/null || pgrep -f "run_esci.py" >/dev/null || pgrep -f "03_run_model.py" >/dev/null; do sleep 20; done; }

echo "[phase3] waiting for PHASE2 COMPLETE..."
while ! grep -q "PHASE2 COMPLETE" "$PHASE2_OUT" 2>/dev/null; do sleep 20; done
wait_clear
echo "[phase3] starting ESCI probe + schema ablation."

TAGS=(qwen4 llama4 mistral4 gemma4)
DIRS=(
  "models--mlx-community--Qwen2.5-7B-Instruct-4bit"
  "models--mlx-community--Meta-Llama-3.1-8B-Instruct-4bit"
  "models--mlx-community--Mistral-7B-Instruct-v0.3-4bit"
  "models--mlx-community--gemma-2-9b-it-4bit"
)
for i in "${!TAGS[@]}"; do
  m="${TAGS[$i]}"; path=$(echo "$hub/${DIRS[$i]}"/snapshots/*/)
  # second-domain ESCI entity probe (expansion)
  if [ ! -s "results/probe-esci-${m}-exp__esci_probe_metrics.json" ]; then
    wait_clear
    echo "=== ESCI-PROBE $m ==="
    python scripts/run_probe.py --model "$path" --tag "probe-esci-${m}-exp" --mode expansion --probe data/probe_esci/probe.jsonl 2>&1 | tail -1
    python scripts/validate_esci_probe.py --tag "probe-esci-${m}-exp" 2>&1 | tail -3
  fi
  # schema-nullability ablation on WANDS probe (exp_optional)
  if [ ! -s "results/probe-${m}-opt__probe_metrics.json" ]; then
    wait_clear
    echo "=== SCHEMA-ABLATION $m (exp_optional) ==="
    python scripts/run_probe.py --model "$path" --tag "probe-${m}-opt" --mode exp_optional 2>&1 | tail -1
    python scripts/score_probe.py --tag "probe-${m}-opt" 2>&1 | grep -E "type-less|excluded"
    python scripts/validate_retrieval.py --tag "probe-${m}-opt" 2>&1 | grep "REAL recall"
  fi
done
echo "=== PHASE3 COMPLETE ==="
