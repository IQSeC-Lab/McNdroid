nohup python -u mcndroid_cl_class_inc.py \
  --dataset_root ./mcndroid_malware_class_inc_dataset \
  --modalities fusion \
  --strategies ER \
  --num_seeds 1 \
  --epochs 30 \
  --batch_size 512 \
  --lr 0.001 \
  --replay_buffer_sizes 2000 \
  --replay_ratio 0.5 \
  --out_dir ./mcndroid_class_inc_output > cl_class_inc_out_ER_fusion.log 2>&1 &

# nohup python -u mcndroid_cl_class_inc.py \
#   --dataset_root ./mcndroid_malware_class_inc_dataset \
#   --modalities data gml json fusion \
#   --strategies None Joint ER \
#   --num_seeds 1 \
#   --epochs 30 \
#   --batch_size 512 \
#   --lr 0.001 \
#   --replay_buffer_sizes 2000 \
#   --replay_ratio 0.5 \
#   --out_dir ./mcndroid_class_inc_output > cl_class_inc_out.log 2>&1 &

# nohup python -u mcndroid_cl_class.py \
#   --dataset_root ./mcndroid_malware_class_inc_dataset \
#   --modalities data gml json fusion \
#   --strategies None Joint ER \
#   --num_seeds 2 \
#   --epochs 30 \
#   --batch_size 512 \
#   --lr 0.001 \
#   --replay_buffer_sizes 1000 2000 \
#   --replay_ratio 0.5 \
#   --out_dir ./mcndroid_class_inc_training_output > cl_class_inc.log 2>&1 &



# python mcndroid_class_inc.py \
#   --data_root ./McNdroid/data_feature/processed_data \
#   --gml_root ./McNdroid/gml_feature/processed_data \
#   --json_root ./McNdroid/json_feature/processed_data \
#   --metadata_csv ./McNdroid/metadata.csv \
#   --train_year 2013 \
#   --test_year 2013 \
#   --n_families 100 \
#   --families_per_step 10 \
#   --modalities data gml json fusion \
#   --out_dir ./cl_class_inc_output > cl_class_inc_output.log 2>&1 &