#!/usr/bin/env bash
# Generation driver for the scaled probe (Option B). Generates plans only; scoring is a separate pass.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
set -a; source ../GroundLM/.env; set +a
export PYTHONHASHSEED=0
hub="$HOME/.cache/huggingface/hub"
LOC_TAGS=(qwen4 llama4 mistral4 gemma4)
LOC_DIRS=("models--mlx-community--Qwen2.5-7B-Instruct-4bit" "models--mlx-community--Meta-Llama-3.1-8B-Instruct-4bit" "models--mlx-community--Mistral-7B-Instruct-v0.3-4bit" "models--mlx-community--gemma-2-9b-it-4bit")
# API: tag|provider|model
API=(
"haiku|anthropic|claude-haiku-4-5-20251001"
"sonnet|anthropic|claude-sonnet-4-6"
"together-llama70|together|meta-llama/Llama-3.3-70B-Instruct-Turbo"
"gem3flash|gemini|gemini-3-flash-preview"
"gem31pro|gemini|gemini-3.1-pro-preview"
"gpt4o|openai|gpt-4o"
"gpt52|openai|gpt-5.2"
)
modes(){ echo "conservative:con expansion:exp exp_optional:opt"; }
done_file(){ [ -s "results/$1__probe_plans.jsonl" ] && [ "$(wc -l < results/$1__probe_plans.jsonl)" -ge "$2" ]; }

echo "===== WANDS-55: all 11 models x 3 modes ====="
NW=$(wc -l < data/probe/probe.jsonl)
for i in "${!LOC_TAGS[@]}"; do
  m="${LOC_TAGS[$i]}"; path=$(echo "$hub/${LOC_DIRS[$i]}"/snapshots/*/)
  for ms in $(modes); do mode="${ms%%:*}"; s="${ms##*:}"; tag="probe-${m}-${s}"
    done_file "$tag" "$NW" && { echo "[skip] $tag"; continue; }
    echo "=== $tag (local) ==="; python scripts/run_probe.py --model "$path" --tag "$tag" --mode "$mode" --probe data/probe/probe.jsonl 2>&1 | tail -1
  done
done
for spec in "${API[@]}"; do IFS='|' read -r m prov mod <<< "$spec"
  for ms in $(modes); do mode="${ms%%:*}"; s="${ms##*:}"; tag="probe-${m}-${s}"
    done_file "$tag" "$NW" && { echo "[skip] $tag"; continue; }
    echo "=== $tag ($prov) ==="; python scripts/run_frontier.py --provider "$prov" --model "$mod" --source probe --tag "${m}-${s}" --mode "$mode" 2>&1 | tail -1
  done
done

echo "===== ESCI-348: 4 local x 3 modes ====="
NE=$(wc -l < data/probe_esci/probe.jsonl)
for i in "${!LOC_TAGS[@]}"; do
  m="${LOC_TAGS[$i]}"; path=$(echo "$hub/${LOC_DIRS[$i]}"/snapshots/*/)
  for ms in $(modes); do mode="${ms%%:*}"; s="${ms##*:}"; tag="probe-esci-${m}-${s}"
    done_file "$tag" "$NE" && { echo "[skip] $tag"; continue; }
    echo "=== $tag (local esci) ==="; python scripts/run_probe.py --model "$path" --tag "$tag" --mode "$mode" --probe data/probe_esci/probe.jsonl 2>&1 | tail -1
  done
done
echo "===== SCALE_RUN COMPLETE ====="
