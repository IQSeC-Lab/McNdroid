# Disagreement Analysis Commands

## 1. Aggregate Disagreement Metrics Across Seeds — aggregate_seeds.py

```bash
python aggregate_seeds.py
```

**Output:**
* `disagreement_results/disagreement_summary_all_seeds.csv`

---

## 2. Compute Disagreement Metrics — disagreement_metrics.py

```bash
python disagreement_metrics.py
```

**Output:**
* `disagreement_results/seed_<seed>/disagreement_<year>.csv`
* `disagreement_results/seed_<seed>/disagreement_summary.csv`
* `disagreement_results/disagreement_summary_all_seeds.csv`

---

## 3. Train XGBoost Models & Generate Prediction Logs: Single Year — modal_disagreement_single.py

```bash
python modal_disagreement_single.py
```

**Output:**
* `saved_models/seed_<seed>/<modality>_xgboost_2013.json`
* `prediction_logs/seed_<seed>/<modality>_predictions_<year>.csv`

---

## 4. Train XGBoost Models & Generate Prediction Logs: Multi Year — modal_disagreement_multi.py

```bash
python modal_disagreement_multi.py
```

**Output:**
* `saved_models/seed_<seed>/<modality>_xgboost_2013.json`
* `prediction_logs/seed_<seed>/<modality>_predictions_<year>.csv`

---

## 5. Identify Modality Dissenter Per Sample — modality_dissenter.py

```bash
python modality_dissenter.py
```

**Output:**
* `disagreement_results/seed_<seed>/modality_dissenter_<year>.csv`

---