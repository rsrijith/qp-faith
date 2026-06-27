#!/usr/bin/env bash
# Full parallel pipeline: all 11 models on BOTH WANDS and Amazon/ESCI probes.
# 4 API provider-streams (threaded, concurrency 8) run concurrently and overlap the
# GPU-bound ESCI-local sweep. Resumable via skip-logic. Then score everything.
cd "$(dirname "$0")/.."; source .venv/bin/activate
set -a; source ../GroundLM/.env; set +a; export PYTHONHASHSEED=0
hub="$HOME/.cache/huggingface/hub"
NW=$(wc -l < data/probe/probe.jsonl); NE=$(wc -l < data/probe_esci/probe.jsonl)

api_run(){  # $1=m $2=provider $3=model $4=src(wands|esci) ; remaining = mode:short pairs
  local m=$1 prov=$2 mod=$3 src=$4; shift 4
  for ms in "$@"; do local mode="${ms%%:*}" s="${ms##*:}"
    if [ "$src" = esci ]; then local tag="probe-esci-${m}-${s}" ftag="esci-${m}-${s}" N=$NE pf="--probe-file data/probe_esci/probe.jsonl"
    else local tag="probe-${m}-${s}" ftag="${m}-${s}" N=$NW pf=""; fi
    if [ -s "results/${tag}__probe_plans.jsonl" ] && [ "$(wc -l < results/${tag}__probe_plans.jsonl)" -ge "$N" ]; then echo "[skip] $tag"; continue; fi
    echo "=== $tag ($prov) ==="
    python scripts/run_frontier.py --provider "$prov" --model "$mod" --source probe $pf --tag "$ftag" --mode "$mode" --concurrency 8 2>&1 | tail -1
  done
}
CE="conservative:con expansion:exp"; CEO="conservative:con expansion:exp exp_optional:opt"

# --- 4 API provider-streams in parallel (WANDS-remaining then ESCI) ---
{ api_run gem3flash gemini gemini-3-flash-preview wands $CE
  api_run gem31pro  gemini gemini-3.1-pro-preview wands $CE
  api_run gem3flash gemini gemini-3-flash-preview esci  $CE
  api_run gem31pro  gemini gemini-3.1-pro-preview esci  $CE ; } &
PA=$!
{ api_run gpt4o openai gpt-4o wands $CEO
  api_run gpt52 openai gpt-5.2 wands $CEO
  api_run gpt4o openai gpt-4o esci  $CE
  api_run gpt52 openai gpt-5.2 esci  $CE ; } &
PB=$!
{ api_run haiku  anthropic claude-haiku-4-5-20251001 esci $CE
  api_run sonnet anthropic claude-sonnet-4-6        esci $CE ; } &
PC=$!
{ api_run together-llama70 together meta-llama/Llama-3.3-70B-Instruct-Turbo esci $CE ; } &
PD=$!

# --- ESCI-local (GPU, serial) in foreground, overlapping the API streams ---
LOC=(qwen4 llama4 mistral4 gemma4); DIRS=("models--mlx-community--Qwen2.5-7B-Instruct-4bit" "models--mlx-community--Meta-Llama-3.1-8B-Instruct-4bit" "models--mlx-community--Mistral-7B-Instruct-v0.3-4bit" "models--mlx-community--gemma-2-9b-it-4bit")
for i in "${!LOC[@]}"; do m="${LOC[$i]}"; path=$(echo "$hub/${DIRS[$i]}"/snapshots/*/)
  for ms in conservative:con expansion:exp exp_optional:opt; do mode="${ms%%:*}"; s="${ms##*:}"; tag="probe-esci-${m}-${s}"
    if [ -s "results/${tag}__probe_plans.jsonl" ] && [ "$(wc -l < results/${tag}__probe_plans.jsonl)" -ge "$NE" ]; then echo "[skip] $tag"; continue; fi
    echo "=== $tag (esci local) ==="; python scripts/run_probe.py --model "$path" --tag "$tag" --mode "$mode" --probe data/probe_esci/probe.jsonl 2>&1 | tail -1
  done
done
wait $PA $PB $PC $PD
echo "===== ALL GENERATION DONE; SCORING ====="
bash scripts/score_all.sh
echo "===== PIPELINE_COMPLETE ====="
