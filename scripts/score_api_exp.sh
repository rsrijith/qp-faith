cd "$(dirname "$0")/.."; source .venv/bin/activate
CAT="data/probe_esci/catalog_full.parquet"; j=0
for m in haiku sonnet together-llama70 gpt4o gpt52; do
  tag="probe-esci-${m}-exp"
  python scripts/validate_esci_probe.py --tag "$tag" --catalog "$CAT" >/dev/null 2>&1 &
  j=$((j+1)); [ $((j%3)) -eq 0 ] && wait
done; wait
echo "API_EXP_SCORED"
