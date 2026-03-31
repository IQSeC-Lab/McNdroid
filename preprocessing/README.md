# Data(DREBIN) Preparation Commands

## 1. Initializer (2013) — WITHOUT VarianceThreshold

```bash
python data_feature_builder.py \
  --mode initializer \
  --data-root /scratch/mkamol/mcndroid/all_data/ \
  --year 2013 \
  --output-dir /scratch/mkamol/datasets/processed_data/2013 \
  --test-size 0.2 \
  --seed 42 \
  --stratify
```

**Output:**

* `train_X_y.npz`
* `test_X_y.npz`
* `vocab.json`

---

## 2. Initializer (2013) — WITH VarianceThreshold

```bash
python data_feature_builder.py \
  --mode initializer \
  --data-root /scratch/mkamol/mcndroid/all_data/ \
  --year 2013 \
  --output-dir /scratch/mkamol/datasets/processed_data/2013 \
  --test-size 0.2 \
  --seed 42 \
  --stratify \
  --use-variance-threshold \
  --variance-threshold 0.001
```

**Output:**

* `train_X_y.npz`
* `test_X_y.npz`
* `vocab.json`
* `selector_meta.json`

---

## 3. Adaptation (e.g., 2014) — WITHOUT VarianceThreshold

```bash
python data_feature_builder.py \
  --mode adaptation \
  --data-root /data/mcndroid/all_data \
  --year 2014 \
  --output-dir /data/mcndroid/processed/2014 \
  --vocab-path /data/mcndroid/processed/2013/vocab.json \
  --test-size 0.2 \
  --seed 42 \
  --stratify
```

**Notes:**

* Uses frozen vocabulary from 2013
* No feature selection applied

---

## 4. Adaptation (e.g., 2014) — WITH VarianceThreshold

```bash
python data_feature_builder.py \
  --mode adaptation \
  --data-root /data/mcndroid/all_data \
  --year 2014 \
  --output-dir /data/mcndroid/processed/2014 \
  --vocab-path /data/mcndroid/processed/2013/vocab.json \
  --selector-meta-path /data/mcndroid/processed/2013/selector_meta.json \
  --test-size 0.2 \
  --seed 42 \
  --stratify
```
# GML Preparation Commands

```bash
python gml_feature_builder.py initializer \
  --year 2013 \
  --out-dir /scratch/mkamol/datasets/init_2013 \
  --workers 16
```
```bash
python gml_feature_builder.py adaptation \
  --year 2014 \
  --init-dir /scratch/mkamol/datasets/init_2013 \
  --out-dir /scratch/mkamol/datasets/adapt_2014_from_2013 \
  --workers 16
```
---
