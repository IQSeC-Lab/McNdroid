## Dataset Distribution and Structure

### Zenodo Download Files

To accommodate Zenodo's file-count constraints, the three modalities are
distributed as separate compressed archives rather than as individual files.
The Zenodo record contains:

```text
mcn_data.zip
mcn_gml.zip
mcn_json.zip
metadata.csv
vendor_family_wide_verdict.csv
```

The archives correspond to the following dataset components:

| Download file | Extracted directory | Content |
|---|---|---|
| `mcn_data.zip` | `data_feature/` | Static DREBIN feature matrices and associated metadata |
| `mcn_gml.zip` | `gml_feature/` | API call graphs in GML format |
| `mcn_json.zip` | `json_feature/` | Dynamic behavioral reports in JSON format |
| `metadata.csv` | — | Dataset-level sample metadata |
| `vendor_family_wide_verdict.csv` | — | Per-vendor malware-family verdicts |

The ZIP files are packaging artifacts used for dataset distribution. The
directory tree below describes the dataset after the archives have been
extracted.

### Extraction

Download all five files into the same directory and extract the three
archives:

```bash
unzip mcn_data.zip
unzip mcn_gml.zip
unzip mcn_json.zip
```

### Extracted Dataset Layout

After extraction, the dataset has the following logical structure:

```text
McNdroid/
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
│           │   ├── train_X.npz
│           │   ├── test_X.npz
│           │   ├── train_meta.npz
│           │   ├── test_meta.npz
│           │   └── split_meta.json
│           ├── ...
│           └── 2025/
├── gml_feature/
│   └── processed_data/
│       └── ...
└── json_feature/
    └── processed_data/
        └── ...
```

The `init_2013` directory indicates that the static feature vocabulary and
feature-selection configuration were constructed from the initial 2013
training partition and reused for subsequent years. Accordingly,
`vocab.json` and `selector_meta.json` are stored in the `2013/` directory,
whereas each yearly directory contains its corresponding feature matrices,
sample metadata, and split metadata.
