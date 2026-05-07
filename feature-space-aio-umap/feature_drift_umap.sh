#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT="${SCRIPT:-mm_feature_Drift.py}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-512}"
TRAIN_YEAR="${TRAIN_YEAR:-2013}"
TEST_YEARS="${TEST_YEARS:-2014 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025}"
GPU_IDS="${GPU_IDS:-}"

MODALITIES=(gml json data)
ROOTS=(
  "gml_feature/processed_data/init_2013"
  "json_feature/processed_data/init_2013"
  "data_feature/processed_data/init_2013"
)
OUT_DIRS=(
  "results/drift_analysis_gml"
  "results/drift_analysis_json"
  "results/drift_analysis_data"
)

mkdir -p logs

echo "Using script: ${SCRIPT}"
echo "Train year: ${TRAIN_YEAR}"
echo "Test years: ${TEST_YEARS}"
echo "Epochs: ${EPOCHS} | Batch size: ${BATCH_SIZE}"

declare -a PIDS=()

auto_gpu="false"
if [[ -n "${GPU_IDS}" ]]; then
  read -r -a GPU_ARR <<< "${GPU_IDS}"
  if [[ "${#GPU_ARR[@]}" -ne 3 ]]; then
    echo "ERROR: GPU_IDS must contain exactly 3 entries, e.g. GPU_IDS='0 1 2'" >&2
    exit 1
  fi
  auto_gpu="true"
fi

for i in "${!MODALITIES[@]}"; do
  modality="${MODALITIES[$i]}"
  root="${ROOTS[$i]}"
  out_dir="${OUT_DIRS[$i]}"
  log_file="logs/${modality}_$(date +%Y%m%d_%H%M%S).log"

  cmd=(
    "${PYTHON_BIN}" "${SCRIPT}"
    --modality "${modality}"
    --root "${root}"
    --out-dir "${out_dir}"
    --train-year "${TRAIN_YEAR}"
    --test-years ${TEST_YEARS}
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --make-all-year-overlays
    --point-size 24
  )

  echo "Launching ${modality} -> ${log_file}"

  if [[ "${auto_gpu}" == "true" ]]; then
    gpu_id="${GPU_ARR[$i]}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}" > "${log_file}" 2>&1
  else
    "${cmd[@]}" > "${log_file}" 2>&1
  fi

  echo "  Finished ${modality}"
done

echo
echo "Started all three modality jobs in parallel."
echo "PIDs: ${PIDS[*]}"
echo "Logs: logs/"
echo

echo "To monitor:"
echo "  tail -f logs/gml_*.log"
echo "  tail -f logs/json_*.log"
echo "  tail -f logs/data_*.log"
echo

echo "To stop all:"
echo "  kill ${PIDS[*]}"
