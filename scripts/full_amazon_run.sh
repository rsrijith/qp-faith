#!/usr/bin/env bash
# Full Amazon/ESCI run: 11 models (4 local + Flash + Flash Lite + Anthropic2 + OpenAI2 + Together)
# on the 1.2M-product probe_full (Gemini Pro dropped: 250 RPD). Tags overwrite the 348-ESCI
# (full supersedes). Harm scored against catalog_full. Resumable via skip-logic. Then score.
cd "$(dirname "$0")/.."; source .venv/bin/activate
set -a; source ../GroundLM/.env; set +a; export PYTHONHASHSEED=0
hub="$HOME/.cache/huggingface/hub"
PF="data/probe_esci/probe_full.jsonl"; CAT="data/probe_esci/catalog_full.parquet"
N=$(wc -l < "$PF")
api_run(){ local m=$1 prov=$2 mod=$3; shift 3
  for ms in "$@"; do local mode="${ms%%:*}" s="${ms##*:}" tag="probe-esci-${m}-${s}"
    if [ -s "results/${tag}__probe_plans.jsonl" ] && [ "$(wc -l < results/${tag}__probe_plans.jsonl)" -ge "$N" ]; then echo "[skip] $tag"; continue; fi
    echo "=== $tag ($prov) ==="
    python scripts/run_frontier.py --provider "$prov" --model "$mod" --source probe --probe-file "$PF" --tag "esci-${m}-${s}" --mode "$mode" --concurrency 8 2>&1 | tail -1
  done
}
CE="conservative:con expansion:exp"
{ api_run gem3flash      gemini    gemini-3-flash-preview  $CE
  api_run gem25flashlite gemini    gemini-2.5-flash-lite   $CE ; } & PA=$!
{ api_run gpt4o openai gpt-4o $CE ; api_run gpt52 openai gpt-5.2 $CE ; } & PB=$!
{ api_run haiku anthropic claude-haiku-4-5-20251001 $CE ; api_run sonnet anthropic claude-sonnet-4-6 $CE ; } & PC=$!
{ api_run together-llama70 together meta-llama/Llama-3.3-70B-Instruct-Turbo $CE ; } & PD=$!
# local (GPU serial) foreground, overlaps API streams
LOC=(qwen4 llama4 mistral4 gemma4); DIRS=("models--mlx-community--Qwen2.5-7B-Instruct-4bit" "models--mlx-community--Meta-Llama-3.1-8B-Instruct-4bit" "models--mlx-community--Mistral-7B-Instruct-v0.3-4bit" "models--mlx-community--gemma-2-9b-it-4bit")
for i in "${!LOC[@]}"; do m="${LOC[$i]}"; path=$(echo "$hub/${DIRS[$i]}"/snapshots/*/)
  for ms in conservative:con expansion:exp; do mode="${ms%%:*}"; s="${ms##*:}"; tag="probe-esci-${m}-${s}"
    if [ -s "results/${tag}__probe_plans.jsonl" ] && [ "$(wc -l < results/${tag}__probe_plans.jsonl)" -ge "$N" ]; then echo "[skip] $tag"; continue; fi
    echo "=== $tag (local) ==="; python scripts/run_probe.py --model "$path" --tag "$tag" --mode "$mode" --probe "$PF" 2>&1 | tail -1
  done
done
wait $PA $PB $PC $PD
echo "===== FULL-AMAZON GENERATION DONE; SCORING ====="
for m in qwen4 llama4 mistral4 gemma4 haiku sonnet together-llama70 gem3flash gem25flashlite gpt4o gpt52; do
  for s in con exp; do tag="probe-esci-${m}-${s}"
    [ -s "results/${tag}__probe_plans.jsonl" ] || continue
    python scripts/score_probe.py --tag "$tag" >/dev/null 2>&1
    python scripts/validate_esci_probe.py --tag "$tag" --catalog "$CAT" >/dev/null 2>&1
    echo "scored $tag"
  done
done
echo "===== FULL_AMAZON_COMPLETE ====="
