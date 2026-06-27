#!/usr/bin/env bash
set -u; cd "$(dirname "$0")/.."; source .venv/bin/activate
WANDS=(qwen4 llama4 mistral4 gemma4 haiku sonnet together-llama70 gem3flash gem31pro gpt4o gpt52)
ESCI=(qwen4 llama4 mistral4 gemma4 haiku sonnet together-llama70 gem3flash gem31pro gpt4o gpt52)
echo "===== SCORING WANDS ====="
for m in "${WANDS[@]}"; do for s in con exp opt; do tag="probe-${m}-${s}"
  [ -s "results/${tag}__probe_plans.jsonl" ] || { echo "[miss] $tag"; continue; }
  python scripts/score_probe.py --tag "$tag" >/dev/null 2>&1
  [ "$s" != opt ] && python scripts/validate_retrieval.py --tag "$tag" >/dev/null 2>&1
  echo "scored $tag"
done; done
echo "===== SCORING ESCI ====="
for m in "${ESCI[@]}"; do for s in con exp opt; do tag="probe-esci-${m}-${s}"
  [ -s "results/${tag}__probe_plans.jsonl" ] || { echo "[miss] $tag"; continue; }
  python scripts/score_probe.py --tag "$tag" >/dev/null 2>&1
  [ "$s" != opt ] && python scripts/validate_esci_probe.py --tag "$tag" >/dev/null 2>&1
  echo "scored $tag"
done; done
echo "===== SCORE_ALL COMPLETE ====="
