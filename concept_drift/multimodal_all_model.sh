#!/bin/bash

set -e

SCRIPT="mm_all_model.py"

TRAIN_YEAR=2013
TEST_START_YEAR=2013
TEST_END_YEAR=2025
SKIP_YEARS="2015"

STAGE="all"
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <out_base>"
  exit 1
fi

OUT_BASE="$1"

MAX_PARALLEL=6

MODELS=(
  "xgboost"
  "lightgbm"
  "svm"
  "mlp"
  "detectbert"
  "vit"
)

SEEDS=(
  137
  491
)

mkdir -p "$OUT_BASE/logs"

run_model() {
  MODEL="$1"
  SEED="$2"

  echo "Starting model: $MODEL (seed=$SEED)"

  python3 "$SCRIPT" \
    --stage "$STAGE" \
    --model "$MODEL" \
    --seed "$SEED" \
    --out-dir "$OUT_BASE/seed_$SEED/$MODEL/" \
    > "$OUT_BASE/logs/${MODEL}_seed_${SEED}.log" 2>&1

  echo "Finished model: $MODEL (seed=$SEED)"
}

for MODEL in "${MODELS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    run_model "$MODEL" "$SEED" &

    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
      sleep 10
    done
  done
done

wait

echo "All models completed."
echo "Results saved under: $OUT_BASE"
echo "Logs saved under: $OUT_BASE/logs"