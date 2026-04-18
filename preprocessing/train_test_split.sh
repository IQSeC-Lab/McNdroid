for year in {2013..2025}; do
  if [ "$year" -eq 2015 ]; then
    continue
  fi

  python3 train_test_split.py \
    --year "$year" \
    --gml-root /data/mcndroid/jsonl_gml_reports \
    --data-root /data/mcndroid/all_data \
    --json-root /data/mcndroid/all_json \
    --out "splits/${year}_split.json"
done
