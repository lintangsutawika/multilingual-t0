CUR_TIME=`date +%H%M_%m%d%Y`

python /users/zyong2/data/zyong2/mt0/data/external/mt0/multilingual_t0/main.py \
	--model_name_or_path "google/mt5-small" \
	--cache_dir "/users/zyong2/data/zyong2/huggingface/mt5_small" \
	--dataset_name "mc4" \
	--max_input_length 1024 \
	--max_target_length 1024 \
	--do_train \
	--preprocessing_num_workers 4 \
	--per_device_train_batch_size 1 \
	--gradient_accumulation 8 \
	--overwrite_output_dir \
	--output_dir "/users/zyong2/data/zyong2/mt0/data/processed/001/mt5_xxl" \
	--max_steps 100 \
	--per_device_eval_batch_size 1 \
	--do_eval \
	--evaluation_strategy "steps" \
	--eval_steps 10 \
	--logging_dir "/users/zyong2/data/zyong2/mt0/data/processed/001/runs/mt5_xxl_${CUR_TIME}"\
	--logging_strategy "steps" \
	--logging_steps 10 \
	--report_to "tensorboard"

