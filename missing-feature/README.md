# Missing Feature Analysis

This module analyzes the robustness of multimodal malware detection models when certain feature modalities are missing at test time.

## Overview

In real-world scenarios, some feature modalities may be unavailable due to:

- Collection failures
- Incomplete analysis
- Time constraints
- Resource limitations

This module tests how well multimodal fusion models handle missing modalities by:

1. Training on complete multimodal data
2. Evaluating with various modalities deliberately removed
3. Measuring performance degradation under different missing scenarios

The analysis covers both complete modality absence and partial feature masking (randomly removing a percentage of features within a modality).

## Script

### missing_feat.py

Evaluates model robustness under missing feature scenarios using XGBoost-based multimodal fusion.

**Usage:**

```bash
python missing_feat.py \
  --data-root "/path/to/data_feature/processed_data" \
  --gml-root "/path/to/gml_feature/processed_data" \
  --json-root "/path/to/json_feature/processed_data" \
  --train-year 2013 \
  --test-start-year 2013 \
  --test-end-year 2025 \
  --out-dir "./output"
```

**Arguments:**

- `--data-root`: Path to static feature data directory
- `--gml-root`: Path to graph feature data directory
- `--json-root`: Path to dynamic feature data directory
- `--train-year`: Fixed training year (default: 2013)
- `--test-start-year`: First test year (default: 2013)
- `--test-end-year`: Last test year (default: 2025)
- `--skip-years`: Years to skip (default: "2015")
- `--out-dir`: Output directory for results
- `--random-mask-rates`: Comma-separated rates for random feature masking (e.g., 0.25,0.5,0.75)
- `--json-var-threshold`: Variance threshold for JSON features
- `--threshold`: Classification threshold (default: 0.5)
- `--val-size`: Validation set size fraction
- XGBoost hyperparameters: `--n-estimators`, `--max-depth`, `--learning-rate`, `--subsample`, etc.

## Missing Scenarios

The script tests various missing modality configurations:

- `missing_data`: Static features missing
- `missing_gml`: Graph features missing
- `missing_json`: Dynamic features missing
- `missing_data_gml`: Static + graph features missing
- `missing_data_json`: Static + dynamic features missing
- `missing_gml_json`: Graph + dynamic features missing
- `random_mask_{modality}_{rate}`: Random feature masking at specified rate

## Random Feature Masking

In addition to complete modality absence, the module supports random feature masking within modalities. For example, `--random-mask-rates 0.25,0.5,0.75` will test:

- 25% of features randomly masked
- 50% of features randomly masked
- 75% of features randomly masked

This simulates scenarios where only partial feature information is available.

## Output

The script generates:

- `missing_feature_summary.json`: Global summary of all experiments
- `missing_feature_metrics.json`: Detailed metrics per scenario
- `missing_feature_predictions.npz`: Raw predictions for analysis

Metrics include: accuracy, precision, recall, F1, ROC-AUC, and average precision.

## Shell Wrapper

The folder includes a shell script `missing_feat.sh` that runs the experiment with multiple random seeds for statistical robustness:

```bash
bash missing_feat.sh
```

This runs 3 iterations with different seeds (42, 43, 44) and aggregates results.
