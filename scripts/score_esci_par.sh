#!/usr/bin/env bash
cd "$(dirname "$0")/.."; source .venv/bin/activate
CAT="data/probe_esci/catalog_full.parquet"
# score_probe (fast) for all first
for m in qwen4 llama4 mistral4 gemma4 haiku sonnet together-llama70 gpt4o gpt52; do for s in con exp; do
  python scripts/score_probe.py --tag "probe-esci-${m}-${s}" >/dev/null 2>&1; done; done
# validate_esci (slow, 1.2M) in parallel batches of 3
jobs_run=0
for m in qwen4 llama4 mistral4 gemma4 haiku sonnet together-llama70 gpt4o gpt52; do for s in con exp; do
  tag="probe-esci-${m}-${s}"
  [ -f "results/${tag}__esci_probe_metrics.json" ] && continue
  python scripts/validate_esci_probe.py --tag "$tag" --catalog "$CAT" >/dev/null 2>&1 &
  jobs_run=$((jobs_run+1))
  if [ $((jobs_run % 3)) -eq 0 ]; then wait; fi
done; done
wait
echo "===== ESCI_PAR_DONE ====="
