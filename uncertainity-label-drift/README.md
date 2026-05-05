# Drift Analysis Commands

## 1. Drift Pipeline — drift_pipeline_b2m.py

```bash
python drift_pipeline_b2m.py \
  --drift_data_root ./drift_data \
  --b2m ./label-drift/benign_to_malware.csv \
  --m2b ./label-drift/malware_to_benign.csv \
  --output_dir /home/<user>/2026NeurIPS/drifted_results
```

**Output:**
* `<output_dir>/<year>_budget50.txt`
* `<output_dir>/<year>_budget100.txt`

--- 

## 2. Merge Drift Data — merge_drift_data.py

```bash
python merge_drift_data.py \
  --dataset_root ./dataset \
  --output_dir ./drift_data
```

**Output:**
* `drift_data/data_feature/<year>/merged.npz`
* `drift_data/gml_feature/<year>/merged.npz`
* `drift_data/json_feature/<year>/merged.npz`

---

## 3. Drift vs. Undrift Uncertainty Experiment

```bash
python run_drift_experiment.py
```

**Output:**
* `drift_v_undrift_uncertainty.txt`
* `drift_per_sample.csv`

---
