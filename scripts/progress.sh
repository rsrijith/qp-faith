#!/usr/bin/env bash
cd "$(dirname "$0")/.."
N=$(wc -l < data/probe_esci/probe_full.jsonl)
echo "===== FULL-AMAZON PROGRESS  $(date '+%H:%M:%S')  (target $N motifs/run) ====="
done=0; total=0
echo "-- local (GPU, serial) --"
for m in qwen4 llama4 mistral4 gemma4; do for s in con exp; do
  f="results/probe-esci-${m}-${s}__probe_plans.jsonl"; n=$(wc -l < "$f" 2>/dev/null || echo 0)
  [ "$n" -ge "$N" ] && st="DONE" || st="$n/$N"; total=$((total+1)); [ "$n" -ge "$N" ] && done=$((done+1))
  printf "   %-26s %s\n" "probe-esci-${m}-${s}" "$st"
done; done
echo "-- API (threaded streams) --"
for m in gem3flash gem25flashlite haiku sonnet together-llama70 gpt4o gpt52; do for s in con exp; do
  f="results/probe-esci-${m}-${s}__probe_plans.jsonl"; n=$(wc -l < "$f" 2>/dev/null || echo 0)
  [ "$n" -ge "$N" ] && st="DONE" || st="$n/$N"; total=$((total+1)); [ "$n" -ge "$N" ] && done=$((done+1))
  printf "   %-26s %s\n" "probe-esci-${m}-${s}" "$st"
done; done
echo "-- overall: $done/$total runs complete --"
pgrep -f "run_probe.py|run_frontier.py" >/dev/null && echo "STATUS: running" || echo "STATUS: generation idle (scoring or done)"
grep -q "FULL_AMAZON_COMPLETE" results/full_amazon.log 2>/dev/null && echo "PIPELINE: COMPLETE" || echo "PIPELINE: in progress"
