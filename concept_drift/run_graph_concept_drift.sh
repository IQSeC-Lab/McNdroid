#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_graph_concept_drift.sh 2013

TRAIN_YEAR="${1:-}"

if [[ -z "${TRAIN_YEAR}" ]]; then
  echo "Usage: $0 <train_year>"
  exit 1
fi

ALL_YEARS=(2013 2014 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025)

# Validate train year
FOUND=0
for y in "${ALL_YEARS[@]}"; do
  if [[ "$y" == "$TRAIN_YEAR" ]]; then
    FOUND=1
    break
  fi
done

if [[ "$FOUND" -eq 0 ]]; then
  echo "Error: invalid train year '$TRAIN_YEAR'"
  echo "Allowed years: ${ALL_YEARS[*]}"
  exit 1
fi

SCRIPT="/home/shared-datasets/McNdroid/unimodal_all_model.py"
BASE_DATA_DIR="/home/shared-datasets/McNdroid/gml_feature/processed_data"
DATA_DIR="${BASE_DATA_DIR}/init_${TRAIN_YEAR}"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Error: feature directory does not exist: $DATA_DIR"
  exit 1
fi

# test years = all except selected training year
# TEST_YEARS=()
# for y in "${ALL_YEARS[@]}"; do
#   if [[ "$y" != "$TRAIN_YEAR" ]]; then
#     TEST_YEARS+=("$y")
#   fi
# done
TEST_YEARS=()
for y in "${ALL_YEARS[@]}"; do
  TEST_YEARS+=("$y")
done


MODELS=(
  mlp
  lightgbm
  xgboost
  svm
  detectbert
  vit
)

MODE="graph"
LABEL="binary"
RUN_ID=$((RANDOM % 10000))
MAX_JOBS=3

run_model() {
  local model="$1"

  cmd=(
    python3 "$SCRIPT"
    --mode "$MODE"
    --model "$model"
    --train "$DATA_DIR"
    --test "$DATA_DIR"
    --train-years "$TRAIN_YEAR"
    --test-years "${TEST_YEARS[@]}"
    --label "$LABEL"
    --run "$RUN_ID"
  )

  echo "=================================================="
  echo "Starting model: $model"
  echo "Train year     : $TRAIN_YEAR"
  echo "Feature dir    : $DATA_DIR"
  echo "Test years     : ${TEST_YEARS[*]}"
  printf 'Command        : '
  printf '%q ' "${cmd[@]}"
  echo
  echo "=================================================="

  "${cmd[@]}"

  echo "Finished model: $model"
}

for model in "${MODELS[@]}"; do
  run_model "$model" &

  while [[ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]]; do
    sleep 1
  done
done

wait
echo "All jobs completed."

