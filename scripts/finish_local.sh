#!/usr/bin/env bash
# Clean sequential finish of remaining LOCAL GPU work (one model at a time; no
# self-matching wait_clear). Order: (1) schema-nullability ablation = the FIX PROOF,
# (2) ESCI second-domain entity probe, (3) ESCI brand/color secondary.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONHASHSEED=0
hub="$HOME/.cache/huggingface/hub"
TAGS=(qwen4 llama4 mistral4 gemma4)
DIRS=("models--mlx-community--Qwen2.5-7B-Instruct-4bit" "models--mlx-community--Meta-Llama-3.1-8B-Instruct-4bit" "models--mlx-community--Mistral-7B-Instruct-v0.3-4bit" "models--mlx-community--gemma-2-9b-it-4bit")

echo "### (1) SCHEMA-NULLABILITY ABLATION (fix proof) ###"
for i in "${!TAGS[@]}"; do
  m="${TAGS[$i]}"; path=$(echo "$hub/${DIRS[$i]}"/snapshots/*/)
  [ -s "results/probe-${m}-opt__probe_plans.jsonl" ] && { echo "[skip] $m-opt"; continue; }
  python scripts/run_probe.py --model "$path" --tag "probe-${m}-opt" --mode exp_optional 2>&1 | tail -1
  echo -n "  $m exp_optional: "; python scripts/score_probe.py --tag "probe-${m}-opt" 2>&1 | grep "type-less"
done
echo "### (2) ESCI SECOND-DOMAIN ENTITY PROBE ###"
for i in "${!TAGS[@]}"; do
  m="${TAGS[$i]}"; path=$(echo "$hub/${DIRS[$i]}"/snapshots/*/)
  [ -s "results/probe-esci-${m}-exp__esci_probe_metrics.json" ] && { echo "[skip] esci-$m"; continue; }
  python scripts/run_probe.py --model "$path" --tag "probe-esci-${m}-exp" --mode expansion --probe data/probe_esci/probe.jsonl 2>&1 | tail -1
  python scripts/validate_esci_probe.py --tag "probe-esci-${m}-exp" 2>&1 | grep -E "type-forced|recall"
done
echo "### (3) ESCI BRAND/COLOR SECONDARY (remaining) ###"
for i in "${!TAGS[@]}"; do
  m="${TAGS[$i]}"; path=$(echo "$hub/${DIRS[$i]}"/snapshots/*/)
  for mode in conservative expansion; do
    short=$([ "$mode" = conservative ] && echo con || echo exp)
    [ -s "results/esci-${m}-${short}__scored.parquet" ] && continue
    python scripts/run_esci.py --model "$path" --tag "${m}-${short}" --mode "$mode" 2>&1 | tail -1
    python scripts/score_esci.py --tag "${m}-${short}" 2>&1 | grep "frac-q" || true
  done
done
echo "=== FINISH_LOCAL COMPLETE ==="
