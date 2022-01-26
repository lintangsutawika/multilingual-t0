CUR_TIME=`date +%H%M_%m%d%Y`
MODEL_NAME="google/mt5-xl"
CACHE_DIR="/users/zyong2/data/zyong2/huggingface/mt5_xl"
TRAIN_BSZ=1
GRAD_ACC=16
OUTPUT_DIR="/users/zyong2/data/zyong2/mt0/data/processed/001/mt5_xl"
MAX_STEPS=30
LOGGING_DIR="/users/zyong2/data/zyong2/mt0/data/processed/001/runs/mt5_xl_${CUR_TIME}"
LOGGING_STEPS=10
SAVE_STEPS=10
DS_CONFIG="/users/zyong2/data/zyong2/mt0/data/external/mt0/multilingual_t0/ds_config_zero3.json"


deepspeed \
/users/zyong2/data/zyong2/mt0/data/external/mt0/multilingual_t0/main.py \
--model_name_or_path $MODEL_NAME \
--cache_dir $CACHE_DIR \
--dataset_name "mc4" \
--max_input_length 1024 \
--max_target_length 256 \
--do_train \
--preprocessing_num_workers 4 \
--per_device_train_batch_size $TRAIN_BSZ \
--gradient_accumulation $GRAD_ACC \
--overwrite_output_dir \
--output_dir $OUTPUT_DIR \
--max_steps $MAX_STEPS \
--logging_dir $LOGGING_DIR \
--logging_strategy "steps" \
--logging_steps $LOGGING_STEPS \
--save_strategy "steps" \
--save_steps $SAVE_STEPS \
--report_to "tensorboard" \
--deepspeed $DS_CONFIG

# --per_device_eval_batch_size 1 \
# --do_eval \
# --evaluation_strategy "steps" \
# --eval_steps 10 \