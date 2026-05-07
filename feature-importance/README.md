# Feature Importance

This module analyzes feature importance using SHAP (SHapley Additive exPlanations) values, computing aggregated statistics across feature groups and temporal splits.

## Overview

The module uses XGBoost models trained on malware detection tasks and computes SHAP values to understand which features are most important for predictions. It handles three modalities (static, dynamic, graph-based) and aggregates results across:

- Multiple runs for statistical robustness
- Different temporal splits (IID, NEAR, FAR)
- Feature groups defined in metadata

## Script

### shap-feature-group-mean-std.py

Computes SHAP values and aggregates them by feature group with mean and standard deviation.

**Usage:**

```bash
python shap-feature-group-mean-std.py \
  --data-root "/path/to/data_feature/processed_data" \
  --gml-root "/path/to/gml_feature/processed_data" \
  --json-root "/path/to/json_feature/processed_data" \
  --out-dir "./output"
```

**Arguments:**

- `--data-root`: Path to static feature data directory
- `--gml-root`: Path to graph feature data directory
- `--json-root`: Path to dynamic feature data directory
- `--out-dir`: Output directory for results
- `--data-vocab-json`: Vocabulary JSON for static features
- `--data-selector-json`: Selector metadata JSON
- `--gml-vocab-txt`: Vocabulary file for graph features
- `--json-feature-names-json`: Feature names JSON for dynamic features
- `--n-estimators`: Number of XGBoost estimators (default: 3000)
- `--seed`: Random seed for reproducibility
- `--n-runs`: Number of runs with different random splits (default: 3)
- `--val-size`: Validation set size fraction (default: 0.15)
- `--threshold`: Classification threshold (default: 0.5)
- `--json-var-threshold`: Variance threshold for JSON features (default: 0.0)
- `--top-k-groups`: Number of top feature groups to report (default: 5)
- `--top-k-features-all-years`: Number of top features per year to report (default: 5)

## Temporal Splits

The module analyzes three temporal configurations:

- **IID**: Nearest years (2013, 2014) - assumes near-identical distribution
- **NEAR**: Moderately distant years (2016, 2017)
- **FAR**: Distant years (2018-2025) - maximum temporal distance

## Output

The script produces:

- Feature group importance rankings (mean SHAP values)
- Per-feature importance with standard deviation across runs
- Top-k features for each temporal split
- Statistical summaries of SHAP value distributions

## Feature Groups

Features are grouped according to metadata in the source datasets. Groups can include:

- Permission-based features
- API call features
- Intent-based features
- Graph topology features
- Behavioral features

This allows analysis of which feature categories contribute most to malware detection.
