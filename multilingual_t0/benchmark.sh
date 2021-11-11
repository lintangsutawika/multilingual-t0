T5X_DIR="/home/lintangsutawika/t5x"  # directory where the t5x is cloned, e.g., ${HOME}"/t5x".
TFDS_DATA_DIR="gs://bigscience/experiment_d/multilingual_t0/first-run/data/"
MODEL_DIR="gs://bigscience/experiment_d/multilingual_t0/first-run/model/"

PROJECT_DIR="/home/lintangsutawika/multilingual_t0/multilingual_t0"
export PYTHONPATH=${PROJECT_DIR}

python3.7 ${T5X_DIR}/t5x/train.py \
  --tfds_data_dir=${TFDS_DATA_DIR} \
  --gin_file="mt0_base_finetune.gin" \
  --gin.TRAIN_STEPS=100005 \
  --gin.MODEL_DIR="'${MODEL_DIR}'" \
