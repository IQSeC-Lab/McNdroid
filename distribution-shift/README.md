# Distribution Shift

This module measures distributional shift between training and test data across different years using statistical hypothesis testing.

## Overview

The module uses the two-sample Kolmogorov-Smirnov (KS) test to detect distributional changes in feature distributions between training and test sets. This helps quantify how the data distribution changes over time, which is critical for understanding temporal generalization in malware detection.

The KS test computes the maximum difference between the empirical cumulative distribution functions (ECDFs) of two samples, providing a non-parametric measure of distributional difference.

## Script

### ks-test.py

Performs KS tests between a fixed training year and multiple test years across all modalities.

**Usage:**

```bash
python ks-test.py \
  --data-root "/path/to/data_feature/processed_data/init_2013" \
  --json-root "/path/to/json_feature/processed_data/init_2013" \
  --gml-root "/path/to/gml_feature/processed_data/init_2013" \
  --train-year 2013 \
  --test-start-year 2013 \
  --test-end-year 2025 \
  --out-dir "./output"
```

**Arguments:**

- `--data-root`: Path to static feature data directory
- `--json-root`: Path to dynamic feature data directory
- `--gml-root`: Path to graph feature data directory
- `--train-year`: Year to use for training (reference distribution)
- `--test-start-year`: First year to test against
- `--test-end-year`: Last year to test against
- `--skip-years`: Years to skip (default: "2015")
- `--modalities`: Which modalities to analyze (default: data, gml, json)
- `--max-features`: Optional random feature subsample for high-dimensional data
- `--seed`: Random seed for reproducibility
- `--n-runs`: Number of runs with different feature subsamples
- `--zero-variance-policy`: How to handle zero-variance features (keep/drop)
- `--out-dir`: Output directory for results

## Output

The script generates several CSV files for analysis:

- `{modality}_mean_ks.csv` - Year-by-year mean KS statistic
- `{modality}_mean_ks_mean_std.csv` - Mean and std across runs
- `all_modalities_mean_ks.csv` - Combined results for all modalities

The KS statistic ranges from 0 (identical distributions) to 1 (completely different distributions).

## Supported Modalities

- **data** (static): Static analysis features
- **json** (dynamic): Dynamic analysis features
- **gml** (graph-based): Graph-based features
