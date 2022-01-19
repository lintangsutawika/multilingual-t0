T5X_DIR=${HOME}"/t5x"  # directory where the t5x is cloned, e.g., ${HOME}"/t5x".
MODEL_DIR=$1
EXPERIMENT_NAME=$2

PROJECT_DIR=${HOME}"/multilingual-t0/multilingual_t0/"
export PYTHONPATH=${PROJECT_DIR}

# Logs
LOGS_PATH="/home/lintang/logs"
mkdir -p $LOGS_PATH

python3 ${T5X_DIR}/t5x/train.py \
  --gin_file="mt0_xl_finetune.gin" \
  --gin.MODEL_DIR="'${MODEL_DIR}'" \
  --gin.train.infer_eval_dataset_cfg=None \
  --gin.train.train_eval_dataset_cfg=None \
  --gin.trainer.Trainer.num_microbatches=16 \
  --gin.MIXTURE_OR_TASK_NAME="'t0_plus_plus_train'" \
  2>&1 | tee $LOGS_PATH/train_$EXPERIMENT_NAME.txt
