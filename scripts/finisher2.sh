#!/usr/bin/env bash
cd "$(dirname "$0")/.."; source .venv/bin/activate
set -a; source ../GroundLM/.env; set +a; export PYTHONHASHSEED=0
hub="$HOME/.cache/huggingface/hub"; PF="data/probe_esci/probe_full.jsonl"; CAT="data/probe_esci/catalog_full.parquet"
N=$(wc -l < "$PF")
need(){ ! { [ -s "results/$1__probe_plans.jsonl" ] && [ "$(wc -l < results/$1__probe_plans.jsonl)" -ge "$N" ]; }; }
# --- API: ONE model-mode at a time (reliable), threaded within each ---
{
for spec in "gpt4o|openai|gpt-4o" "gpt52|openai|gpt-5.2" "haiku|anthropic|claude-haiku-4-5-20251001" \
            "sonnet|anthropic|claude-sonnet-4-6" "together-llama70|together|meta-llama/Llama-3.3-70B-Instruct-Turbo" \
            "gem3flash|gemini|gemini-3-flash-preview" "gem25flashlite|gemini|gemini-2.5-flash-lite"; do
  IFS='|' read -r m prov mod <<< "$spec"
  for s in con exp; do mode=$([ "$s" = con ] && echo conservative || echo expansion); tag="probe-esci-${m}-${s}"
    need "$tag" || { echo "[skip] $tag"; continue; }
    echo "=== $tag ($prov) ==="; python scripts/run_frontier.py --provider "$prov" --model "$mod" --source probe --probe-file "$PF" --tag "esci-${m}-${s}" --mode "$mode" --concurrency 8 2>&1 | tail -1
  done
done; echo "API_DONE"; } &
APID=$!
# --- local: GPU serial, foreground ---
LOC=(mistral4 gemma4 qwen4 llama4); DIRS=("models--mlx-community--Mistral-7B-Instruct-v0.3-4bit" "models--mlx-community--gemma-2-9b-it-4bit" "models--mlx-community--Qwen2.5-7B-Instruct-4bit" "models--mlx-community--Meta-Llama-3.1-8B-Instruct-4bit")
for i in "${!LOC[@]}"; do m="${LOC[$i]}"; path=$(echo "$hub/${DIRS[$i]}"/snapshots/*/)
  for s in con exp; do mode=$([ "$s" = con ] && echo conservative || echo expansion); tag="probe-esci-${m}-${s}"
    need "$tag" || { echo "[skip] $tag"; continue; }
    echo "=== $tag (local) ==="; python scripts/run_probe.py --model "$path" --tag "$tag" --mode "$mode" --probe "$PF" 2>&1 | tail -1
  done
done
wait $APID
echo "===== SCORING ====="
for m in qwen4 llama4 mistral4 gemma4 haiku sonnet together-llama70 gem3flash gem25flashlite gpt4o gpt52; do for s in con exp; do tag="probe-esci-${m}-${s}"
  [ -s "results/${tag}__probe_plans.jsonl" ] || continue
  python scripts/score_probe.py --tag "$tag" >/dev/null 2>&1
  python scripts/validate_esci_probe.py --tag "$tag" --catalog "$CAT" >/dev/null 2>&1; echo "scored $tag"
done; done
echo "===== FINISHER2_COMPLETE ====="
