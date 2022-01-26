#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset, load_metric, interleave_datasets
from itertools import islice

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

import math
import random

# import tasks
# from p3 import p3

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.16.0.dev0")

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")

logger = logging.getLogger(__name__)

MT0_LANG_TO_PROBS = {
    'en': 5.67,
    'ru': 3.71,
    'es': 3.09,
    'de': 3.05,
    'fr': 2.89,
    'it': 2.43,
    'pt': 2.36,
    'pl': 2.15,
    'nl': 1.98,
    'tr': 1.93,
    'ja': 1.92,
    'vi': 1.87,
    'id': 1.80,
    'cs': 1.72,
    'zh': 1.67,
    'fa': 1.67,
    'ar': 1.66,
    'sv': 1.61,
    'ro': 1.58,
    'el': 1.54,
    'uk': 1.51,
    'hu': 1.48,
    'da': 1.38,
    'fi': 1.35,
    'no': 1.33,
    'bg': 1.29,
    'hi': 1.21,
    'sk': 1.19,
    'ko': 1.14,
    'th': 1.14,
    'ca': 1.12,
    'ms': 1.09,
    'iw': 1.06,
    'lt': 1.04,
    'sl': 0.95,
    'mr': 0.93,
    'bn': 0.91,
    'et': 0.89,
    'lv': 0.87,
    'az': 0.82,
    'gl': 0.79,
    'cy': 0.76,
    'sq': 0.76,
    'ta': 0.73,
    'sr': 0.72,
    'ne': 0.69,
    'lb': 0.68,
    'hy': 0.65,
    'kk': 0.65,
    'ka': 0.64,
    'mt': 0.64,
    'af': 0.63,
    'fil': 0.62,
    'is': 0.62    
    }

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    seqio_mixture_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the Seqio Mixture (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_input_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=256,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=1000,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the :obj:`decoder_start_token_id`."
            "Useful for multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token "
            "needs to be the target language token.(Usually it is the target language token)"
        },
    )

    def __post_init__(self):

        if self.dataset_name is None and self.seqio_mixture_name is None:
            raise ValueError("Need either a dataset name or a seqio mixture name")

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    print(f"==== Model parameters ====\n {model_args}")
    print(f"==== Data parameters ====\n {data_args}")
    print(f"==== Training/evaluation parameters ====\n {training_args}")
    
    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is expected, e.g. with "
            "`--source_prefix 'translate English to German: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For translation, only JSON files are supported, with one field named "translation" containing two keys for the
    # source and target languages (unless you adapt what follows).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        if data_args.dataset_name == "mc4":
            dataset_list = list()
            probs_list = list()
            for lang, prob in MT0_LANG_TO_PROBS.items():
                dataset_list.append(load_dataset(data_args.dataset_name, lang, split="train", streaming=True, cache_dir=model_args.cache_dir))
                probs_list.append(prob / 100)
                # probs_list.append(0.5)
            raw_datasets = interleave_datasets(dataset_list, probabilities=probs_list, seed=42)

        # raw_datasets = load_dataset(
        #     data_args.dataset_name,
        #     data_args.dataset_config_name,
        #     cache_dir=model_args.cache_dir
        # )

    elif data_args.seqio_mixture_name is not None:
        raw_datasets = load_dataset(
            p3.__file__,
            data_args.seqio_mixture_name
            )

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model.resize_token_embeddings(len(tokenizer))

    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.target_lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.target_lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        # column_names = raw_datasets["train"].column_names
        # print(list(islice(raw_datasets, 1))[0].keys())
        column_names = ['text', 'timestamp', 'url']
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        text = examples['text']
        max_input_length_before_eos = data_args.max_input_length - 2
        tokenized_text = tokenizer.encode_plus(
            text,
            add_special_tokens=False,
            padding=False,
        )

        # following select_random_chunk function 
        # in https://github.com/google-research/text-to-text-transfer-transformer/blob/06a0f54bd02b6222399ccee0107a6f881b030fff/t5/data/preprocessors.py#L2075
        if len(tokenized_text['input_ids']) > max_input_length_before_eos:
            num_segments = int(math.ceil(float(len(tokenized_text['input_ids'])) / float(max_input_length_before_eos)))
            start = max_input_length_before_eos * random.randint(0, num_segments)
            end = min(start + max_input_length_before_eos, len(tokenized_text['input_ids']))
            all_input_ids = tokenized_text['input_ids'][start:end]
        else:
            all_input_ids = tokenized_text['input_ids']
        assert len(all_input_ids) <= max_input_length_before_eos

        if len(all_input_ids) >= 2:
            split_idx = random.randint(1, len(all_input_ids) - 1)
            split_input_ids = all_input_ids[:min(split_idx, data_args.max_input_length - 1)] 
            split_input_ids += [tokenizer.eos_token_id]

            split_label_ids = all_input_ids[split_idx:min(len(all_input_ids), split_idx + data_args.max_target_length - 1)] 
            split_label_ids += [tokenizer.eos_token_id]
            # print("decode:", tokenizer.decode(tokenized_text['input_ids']))
            # print("decode input_ids:", tokenizer.decode(input_ids))
            # print("decode label_ids:", tokenizer.decode(label_ids))
        else:
            split_input_ids = all_input_ids + [tokenizer.eos_token_id]
            split_label_ids = [tokenizer.eos_token_id]
        
        # convert back to text 
        assert len(split_input_ids) <= data_args.max_input_length
        assert len(split_label_ids) <= data_args.max_target_length

        inputs = tokenizer.decode(split_input_ids) 
        targets = tokenizer.decode(split_label_ids)

        # print("🔢 len(all_input_ids)", len(all_input_ids))
        # print("len(split_input_ids)", len(split_input_ids))
        # print("len(split_label_ids)", len(split_label_ids))
        # print("split_label_ids", split_label_ids)
        # print("targets:", targets)

        model_inputs = tokenizer.encode_plus(
            inputs,
            add_special_tokens=False,
            padding=padding,
            max_length=data_args.max_input_length,
            truncation=True
        )

        model_inputs['labels'] = tokenizer.encode(
            targets,
            add_special_tokens=False,
            padding=padding,
            max_length=data_args.max_target_length,
            truncation=True
        )

        
        
        # print(tokenizer.decode(model_inputs['input_ids']))
        # print(tokenizer.decode(model_inputs['labels']))
        # print(model_inputs['labels'])
        # print("🐞 model_inputs", len(model_inputs['input_ids']))
        # print("🐞 model_inputs labels", len(model_inputs['labels']))
        
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            model_inputs['labels'][model_inputs['labels'] == tokenizer.pad_token_id] = -100
        
        return model_inputs

    if training_args.do_train:
        # may need https://github.com/huggingface/datasets/issues/2583
        
        # if "train" not in raw_datasets:
        #     raise ValueError("--do_train requires a train dataset")
        # train_dataset = raw_datasets["train"]
        
        raw_datasets = raw_datasets.with_format("torch")
        train_dataset = raw_datasets
        # print(list(islice(train_dataset, 1))) # [{'text': ..., 'timestamp': ..., 'url': ...}]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            # train_dataset = train_dataset.map(
            #     preprocess_function,
            #     num_proc=data_args.preprocessing_num_workers,
            #     remove_columns=column_names,
            #     load_from_cache_file=not data_args.overwrite_cache,
            #     desc="Padding and Tensorize",
            # )
            train_dataset = train_dataset.map(preprocess_function)

    if training_args.do_eval:
        # max_target_length = data_args.max_target_length
        # if "validation" not in raw_datasets:
        #     raise ValueError("--do_eval requires a validation dataset")
        # eval_dataset = raw_datasets["validation"]
        # if data_args.max_eval_samples is not None:
        #     eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        # with training_args.main_process_first(desc="validation dataset map pre-processing"):
        #     eval_dataset = eval_dataset.map(
        #         preprocess_function,
        #         num_proc=data_args.preprocessing_num_workers,
        #         remove_columns=column_names,
        #         load_from_cache_file=not data_args.overwrite_cache,
        #         desc="Padding and Tensorize",
        #     )
        eval_dataset = train_dataset.take(data_args.max_eval_samples)
        train_dataset = train_dataset.skip(data_args.max_eval_samples)

    if training_args.do_predict:
        max_target_length = data_args.max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Padding and Tensorize",
            )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # Metric
    metric = load_metric("accuracy")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"accuracy": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        logging.info("🚂 start training")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        ### comment out because TypeError: object of type 'TorchIterableDataset' has no len()
        # max_train_samples = (
        #     data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        # )
        # metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print("Done:", trainer.save_state())

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        # max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        # metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    writer.write("\n".join(predictions))

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "translation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()