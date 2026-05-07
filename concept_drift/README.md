# Concept Drift

This module analyzes concept drift in malware detection by training models on one year and evaluating on subsequent years to measure temporal generalization.

## Overview

Concept drift occurs when the underlying data distribution changes over time, causing models trained on historical data to degrade in performance. This module evaluates how well malware detection models generalize across time by:

- Training on a specific year
- Testing on multiple years (including the training year and subsequent years)
- Measuring performance degradation as temporal distance increases

## Scripts

### unimodal_all_model.py

Trains and evaluates single-modality (unimodal) models for concept drift analysis.

**Supported Models:**
- MLP (Multi-Layer Perceptron)
- LightGBM
- XGBoost
- SVM (Support Vector Machine)
- DetectBERT
- ViT (Vision Transformer)

**Supported Modalities:**
- Static (data)
- Dynamic (json)
- Graph-based (gml)

**Example Usage:**
```bash
python unimodal_all_model.py \
  --mode static \
  --model lightgbm \
  --train /path/to/data_feature/processed_data/init_2013 \
  --test /path/to/data_feature/processed_data/init_2013 \
  --train-years 2013 \
  --test-years 2013 2014 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 \
  --label binary \
  --run 1
```

### multimodal_all_model.py

Trains and evaluates multimodal fusion models for concept drift analysis. Combines features from multiple modalities (static, dynamic, graph-based) using early or late fusion strategies.

**Example Usage:**
```bash
python multimodal_all_model.py \
  --train /path/to/data \
  --test /path/to/data \
  --train-years 2013 \
  --test-years 2013 2014 2016 2017 2018 2019 \
  --models mlp lightgbm xgboost \
  --fusion early
```

### multimodal_CA.py

Multimodal concept drift analysis with attention-based fusion mechanisms.

## Shell Scripts

The folder includes wrapper shell scripts for running experiments:

- `run_data_concept_drift.sh` - Run concept drift experiments on static features
- `run_dynamic_concept_drift.sh` - Run concept drift experiments on dynamic features
- `run_graph_concept_drift.sh` - Run concept drift experiments on graph features

**Usage:**
```bash
bash run_data_concept_drift.sh 2013
```

This trains on year 2013 and tests on all available years.

## Key Arguments

- `--mode`: Feature modality (static, dynamic, graph)
- `--model`: Model architecture to use
- `--train`: Path to training data directory
- `--test`: Path to test data directory
- `--train-years`: Year(s) to use for training
- `--test-years`: Year(s) to use for testing
- `--label`: Label type (binary, multiclass)
- `--run`: Run identifier

## Output

The scripts output performance metrics for each train-test year combination:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Average Precision Score

These metrics can be plotted to visualize performance degradation over time, indicating the severity of concept drift.