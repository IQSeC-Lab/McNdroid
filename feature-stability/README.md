# Feature Stability

This module analyzes temporal stability and distributional shift of malware features across years.

## Overview

The feature stability analysis measures how the distribution of malware class features changes over time across three modalities:

- **Static** (data): Static analysis features
- **Dynamic** (json): Dynamic analysis features
- **Graph-based** (gml): Graph-based features

For each modality, it computes per-feature divergence between each year's malware-class distribution and a fixed reference year (default: earliest year). This allows comparison of temporal drift across feature types.

## Scripts

### feature-stability.py

Computes mean per-feature divergence and generates line plots showing temporal drift.

```bash
python feature-stability.py \
  --data-root "/path/to/data_feature/processed_data/init_2013" \
  --json-root "/path/to/json_feature/processed_data/init_2013" \
  --gml-root "/path/to/gml_feature/processed_data/init_2013" \
  --out-dir "." \
  --metric jeffreys \
  --class-filter malware \
  --preprocess zscore
```

**Arguments:**

- `--data-root`, `--json-root`, `--gml-root`: Paths to feature data directories
- `--out-dir`: Output directory for plots
- `--metric`: Divergence metric (jeffreys, js, kl)
- `--class-filter`: Class to analyze (default: malware)
- `--preprocess`: Preprocessing method (zscore, standard, minmax)
- `--years`: List of years to analyze (default: 2013-2025)
- `--ref-year`: Reference year for divergence calculation

### malware-family-stability.py

Analyzes drift in malware family distributions across years. Computes entropy-based divergence metrics for each family.

```bash
python malware-family-stability.py
```

The script uses a CSV file (`final_hash_date_label_family.csv`) containing hash, date, label, and family information.

## Notebooks

- `plotter-feature-stability.ipynb`: Visualization of feature stability results
- `plotter-family-stability.ipynb`: Visualization of family stability results

## Metrics

The divergence is computed using information-theoretic measures:

- **Jeffreys divergence**: Symmetric measure based on KL divergence
- **Jensen-Shannon divergence**: Bounded, symmetric divergence
- **KL divergence**: Asymmetric Kullback-Leibler divergence

## Output

The scripts generate line plots showing:

- X-axis: Years
- Y-axis: Mean per-feature divergence
- Multiple lines for different modalities (static, dynamic, graph-based)
