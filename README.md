# mT0: Multilingual T0

Multilingual extension for multitask prompted training.

## mT5 + LM Adaptation
- From the [Google's repo](https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md#lm-adapted-t511lm100k) description, the "LM adapted" models are trained for an additional 100K steps on the LM objective discussed in the T5 paper.
- In T5 paper section 3.2.3, the LM objective is to sample a span of text from the unlabeled data set and choose a random point to split it into prefix and target portions. 
- T5: 100K Steps = 100,000 Steps * 2048 Batch Size * 512 Length = **104,857,600,000 Tokens** with [`pack_examples`](https://github.com/tensorflow/tensor2tensor/blob/2a33b152d7835af66a6d20afe7961751047e28dd/tensor2tensor/data_generators/generator_utils.py#L597) to "pack" multiple sequences into each entry of the batch.

### Training Details
- Model: 
- Tokenizer: mT5 (have 250,000 wordpieces, character coverage of 0.99999, and “byte-fallback” feature)
- Language Sampling: probability p(L) ∝ |L|^α where α = 0.3
  - Use the [`datasets.interleave_datasets`](https://huggingface.co/docs/datasets/_modules/datasets/combine.html#interleave_datasets) function and `streaming` mode 
  - Set probabilities according to Table 6 in [mT5 paper](https://arxiv.org/pdf/2010.11934.pdf)
- Batch size:
- Sequence length:
- Steps: 
- Target length: 256
- Dropout: None

## Installation

This repository uses the following external repositories that need to be setup:
1. [T5X](https://github.com/google-research/t5x)
2. [Promptsource](https://github.com/bigscience-workshop/promptsource)

## Finetuning

The finetuning script is `multilingual_t0/first_run.sh`. The script is as follows.

```
# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.
MODEL_DIR="..."

# Data dir to save the processed dataset in "gs://data_dir" format.
TFDS_DATA_DIR="..."
T5X_DIR="..."  # directory where the T5X repo is cloned.

python3 ${T5X_DIR}/t5x/train.py \
  --tfds_data_dir=${TFDS_DATA_DIR} \
  --gin_file="t5x/examples/t5/t5_1_1/xxl.gin" \
  --gin_file="t5x/configs/runs/finetune.gin" \
  --module_import="multilingual_t0.tasks" \
  --gin.MIXTURE_OR_TASK_NAME="'mt5_d4_gpt_sglue_train'" \
  --gin.MIXTURE_OR_TASK_MODULE="'t5.data.mixtures'" \
  --gin.TASK_FEATURE_LENGTHS="{'inputs': 256, 'targets': 256}" \
  --gin.TRAIN_STEPS=1_125_000 \
  --gin.MODEL_DIR="'${MODEL_DIR}'" \
  --gin.USE_CACHED_TASKS=False \
  --gin.INITIAL_CHECKPOINT_PATH="'gs://t5-data/pretrained_models/mt5/xxl/model.ckpt-1000000'"
```

