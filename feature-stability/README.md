# Stability

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
