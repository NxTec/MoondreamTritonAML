#!/usr/bin/env bash
# Set your Azure ML API key as an environment variable:
# export AZURE_ML_API_KEY="your-api-key-here"
API="${AZURE_ML_API_KEY}"
MAX=256


python test_concurrency.py --api-key "$API" \
  --modes query \
  --concurrency 200 \
  --duration 0 \
  --stop-at-first-failure \
  > conc_aml/query_200.txt 2>&1
#
#for M in detect caption query; do
#  echo "=== MODE: $M ==="
#  python test_concurrency.py --api-key "$API" \
#    --modes "$M" --sweep --max-concurrency "$MAX" \
#    --stop-at-first-failure --duration 0 \
#  | tee "conc_aml/${M}_scan.txt" | tail -n 5
#done

#python test_concurrency.py --api-key "$API" --concurrency 14 --modes detect --duration 0



#python test_concurrency.py --api-key "$API" --max-concurrency 256 \
#  --stop-at-first-failure --duration 60 \
#  --output-json conc_aml/moondream_no_break.json \
#  --output-csv conc_aml/moondream_no_break.csv