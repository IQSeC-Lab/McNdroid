#!/bin/bash

DATA_ROOT="data_feature/processed_data/"
GML_ROOT="gml_feature/processed_data/"
JSON_ROOT="json_feature/processed_data/"
OUT_DIR="results/missing_feature/"

for i in 1 2 3
do
  echo "Run $i started"

  python3 missing_feat.py \
    --data-root "$DATA_ROOT" \
    --gml-root "$GML_ROOT" \
    --json-root "$JSON_ROOT" \
    --out-dir "${OUT_DIR}/run_${i}" \
    --train-year 2013 \
    --test-start-year 2013 \
    --test-end-year 2025 \
    --skip-years 2015 \
    --json-var-threshold 0.001 \
    --threshold 0.5 \
    --seed $((42 + i)) \
    --n-estimators 3000 \
    --max-depth 12 \
    --learning-rate 0.05 \
    --subsample 0.8 \
    --colsample-bytree 0.8 \
    --tree-method hist \
    --xgb-device cuda \
    --n-jobs 8 \
    --random-mask-rates 0.25,0.5,0.75

  echo "Run $i finished"
done