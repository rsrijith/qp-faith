#!/usr/bin/env bash
cd "$(dirname "$0")/.."; source .venv/bin/activate
set -a; source ../GroundLM/.env; set +a; export PYTHONHASHSEED=0
PF="data/probe_esci/probe_full.jsonl"; CAT="data/probe_esci/catalog_full.parquet"
echo "waiting for OpenAI credit..."
for i in $(seq 1 480); do
  if python -c "import os;from openai import OpenAI;OpenAI(api_key=os.environ['OPENAI_API_KEY']).chat.completions.create(model='gpt-4o',max_tokens=3,messages=[{'role':'user','content':'hi'}])" 2>/dev/null; then echo "OPENAI_LIVE (waited ${i}x30s)"; break; fi
  sleep 30
done
# re-run the 3 credit-corrupted OpenAI ESCI runs clean
for spec in "gpt4o|gpt-4o|conservative|con" "gpt4o|gpt-4o|expansion|exp" "gpt52|gpt-5.2|conservative|con"; do
  IFS='|' read -r m mod mode s <<< "$spec"
  echo "=== rerun esci-${m}-${s} ==="
  python scripts/run_frontier.py --provider openai --model "$mod" --source probe --probe-file "$PF" --tag "esci-${m}-${s}" --mode "$mode" --concurrency 8 2>&1 | tail -1
done
# verify OpenAI ESCI now clean
python - <<'PY'
import json
for t in ["probe-esci-gpt4o-con","probe-esci-gpt4o-exp","probe-esci-gpt52-con"]:
    rows=[json.loads(l) for l in open(f"results/{t}__probe_plans.jsonl")]
    e=sum(1 for r in rows if not r.get("plan")); print(f"  {t}: {len(rows)} rows {e} empty ({e/len(rows):.0%})")
PY
echo "===== SCORING ALL 9 ESCI ====="
for m in qwen4 llama4 mistral4 gemma4 haiku sonnet together-llama70 gpt4o gpt52; do for s in con exp; do tag="probe-esci-${m}-${s}"
  python scripts/score_probe.py --tag "$tag" >/dev/null 2>&1
  python scripts/validate_esci_probe.py --tag "$tag" --catalog "$CAT" >/dev/null 2>&1; echo "scored $tag"
done; done
echo "===== ESCI_READY ====="
