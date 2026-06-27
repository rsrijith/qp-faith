#!/usr/bin/env bash
cd "$(dirname "$0")/.."
bash scripts/scale_run.sh
bash scripts/score_all.sh
echo "===== PIPELINE_COMPLETE ====="
