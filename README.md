# McNdroid: A Longitudinal Multimodal Benchmark for Robust Drift Detection in Android Malware

## Dataset

[Zenodo Link](https://zenodo.org/records/19867422?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjZmYWE1MjEyLWRiM2ItNGFhMC05YWVkLTBhM2IzY2YwMTIxYSIsImRhdGEiOnt9LCJyYW5kb20iOiIyNTVhZDIzYWVlYTE2ZDEwZGJjYjVkOTRlODI5YjgwOCJ9.VpaKE5oWc1jV0B84i5wry8QtLoQjii09FD5mi66DaKi5PKCfBmlQkR4neceo_RNQyFcBDHtz_onChFoSyCrwRg)

## Dataset Description

McNdroid is a large-scale, longitudinal, multimodal Android malware detection dataset designed to benchmark concept drift robustness. It spans samples collected from 2013 to 2025 and provides three complementary modalities: static feature vectors, API call graphs (GML), and JSON-based behavioral representations. The dataset also includes a rich metadata CSV and per-vendor family-level verdicts, supporting fine-grained label analysis and multi-label learning.

### Dataset Summary

- **Modalities:** Static features (DREBIN), API call graphs, dynamic behavioral features
- **Time span:** 2013–2025
- **Total size:** ~10.9 GB
- **Splits:** Train/test per year with temporal evaluation protocols
- **Labels:** Binary (malware/benign) and multi-vendor family-level verdicts


### Supported Tasks

- Android malware detection (binary classification)
- Concept drift detection and temporal robustness evaluation
- Multi-modal learning for malware analysis
- Graph-based malware classification


## Dataset Structure

### Repository Layout

```
McNdroid/
├── README.md
├── metadata.csv                         
├── vendor_family_wide_verdict.csv       
├── data_feature/                       
│   └── processed_data/
│       └── init_2013/
│           ├── 2013/
│           │   ├── train_X.npz          
│           │   ├── test_X.npz           
│           │   ├── train_meta.npz       
│           │   ├── test_meta.npz        
│           │   ├── vocab.json           
│           │   ├── selector_meta.json 
│           │   └── split_meta.json      
│           ├── 2014/
│           ├── ...
│           └── 2025/
├── gml_feature/                       
│   └── processed_data/
│       └── ...
├── json_feature/                      
│   └── processed_data/
│       └── ...
```

### Data Fields

#### metadata.csv

Contains per-sample metadata including SHA256 hashes, collection timestamps, labels, and source information.

#### vendor_family_wide_verdict.csv

Contains malware family labels from multiple antivirus vendors, enabling multi-label and label-noise research.

## Dataset Creation

### Source Data

Samples were collected from public malware repositories and benign application stores spanning 2013–2025. Each sample was processed through a static analysis pipeline to extract permissions, API calls, intents, and other manifest and bytecode-level features.

### Annotations

Labels are derived from VirusTotal multi-scanner verdicts. The `vendor_family_wide_verdict.csv` file preserves per-vendor family attributions to support research on label noise and disagreement.
