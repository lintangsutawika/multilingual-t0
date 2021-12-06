T5X_DIR=${HOME}"/t5x"  # directory where the t5x is cloned, e.g., ${HOME}"/t5x".
MODEL_DIR=$1
TFDS_DATA_DIR=$2

PROJECT_DIR=${HOME}"/multilingual-t0/multilingual_t0"
export PYTHONPATH=${PROJECT_DIR}

python3 ${T5X_DIR}/t5x/train.py \
  --tfds_data_dir=${TFDS_DATA_DIR} \
  --gin_file="mt0_xxl_finetune.gin" \
  --gin.MODEL_DIR="'${MODEL_DIR}'" \
  --gin.train.infer_eval_dataset_cfg=None \
  --gin.MIXTURE_OR_TASK_NAME="'d4_train'"
