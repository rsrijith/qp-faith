#!/usr/bin/env bash
cd "$(dirname "$0")/.."; source .venv/bin/activate
CAT="data/probe_esci/catalog_full.parquet"; N=$(wc -l < data/probe_esci/probe_full.jsonl|tr -d ' ')
# wait for gpt4o-exp to finish (up to ~2h), then it's in the set
for i in $(seq 1 240); do n=$(wc -l < results/probe-esci-gpt4o-exp__probe_plans.jsonl 2>/dev/null||echo 0); [ "$n" -ge "$N" ] && break; sleep 30; done
echo "gpt4o-exp final: $(wc -l < results/probe-esci-gpt4o-exp__probe_plans.jsonl 2>/dev/null||echo 0)/$N"
for m in qwen4 llama4 mistral4 gemma4 haiku sonnet together-llama70 gpt4o gpt52; do for s in con exp; do tag="probe-esci-${m}-${s}"
  [ -s "results/${tag}__probe_plans.jsonl" ] || continue
  python scripts/score_probe.py --tag "$tag" >/dev/null 2>&1
  python scripts/validate_esci_probe.py --tag "$tag" --catalog "$CAT" >/dev/null 2>&1
  echo "scored $tag"
done; done
echo "===== ESCI_SCORING_COMPLETE ====="
