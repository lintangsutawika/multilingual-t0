import functools
import seqio
import tensorflow_datasets as tfds
import t5
from t5.evaluation import metrics
from t5.data import preprocessors

import preprocessors

#import csv
#import functools
#from typing import Dict, List, Optional, Tuple

#import datasets
#import pkg_resources
#import seqio
#import tensorflow as tf

#import t5
#from t5.data.glue_utils import get_glue_metric, get_super_glue_metric
from t5.evaluation import metrics as mt

import promptsource.templates
from promptsource.seqio_tasks import utils

from t5x.partitioning import LogicalAxisRules

def fully_sharded_logical_axis_rules() -> LogicalAxisRules:
  """Fully sharded rules for P5X model in terms of logical axes names."""
  return (
      ('batch', 'data'),
      ('vocab', 'model'),
      ('mlp', 'model'),
      ('heads', 'model'),
      ('joined_kv', 'model'),
      ('kv', None),
      ('embed', 'model'),
      ('embed', 'data'),
      ('relpos_buckets', None),
      ('length', None),
      ('layers', None),
      ('stack', None),
  )

#seqio.add_global_cache_dirs(['gs://bigscience/seqio_cached_tasks'])
#seqio.add_global_cache_dirs(['gs://bigscience/experiment_d/experiment_d_cached_tasks/v0.2'])
#seqio.add_global_cache_dirs(['gs://bigscience/experiment_d/multilingual_t0/data/tydi_qa/goldp/3.0.0'])


GET_METRICS = {
    "BLEU": mt.bleu,
    "ROUGE": mt.rouge,
    "Span Squad": mt.span_squad,
    "Squad": mt.squad,
    "Trivia QA": mt.trivia_qa,
    "Accuracy": mt.accuracy,
    "Sequence Accuracy": mt.sequence_accuracy,
    "Pearson Correlation": mt.pearson_corrcoef,
    "Spearman Correlation": mt.spearman_corrcoef,
    "MultiRC": mt.multirc_f1_over_all_answers,
    "AUC": mt.auc,
    "COQA F1": mt.coqa_f1,
    "Edit Distance": mt.edit_distance,
    "Other": mt.accuracy,
}

MAX_EXAMPLES_PER_DATASET = 500_000

MT5_SPM_PATH = "gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model"

MT5_TEMPERATURE = 1.0 / 0.3
MT5_MIX_RATE = functools.partial(
    t5.data.rate_num_examples, temperature=MT5_TEMPERATURE)

MT5_VOCAB = t5.data.SentencePieceVocabulary(MT5_SPM_PATH)
MT5_OUTPUT_FEATURES = {
    "inputs": t5.data.Feature(
        vocabulary=MT5_VOCAB, add_eos=True, required=False),
    "targets": t5.data.Feature(
        vocabulary=MT5_VOCAB, add_eos=True)
}


MAX_EXAMPLES_PER_DATASET = 500_000


def strip_whitespace(output_or_target, example=None, is_target=False):
    """Cached tasks from promptsource all have a leading space on the ground-truth targets."""
    return output_or_target.strip()


def maybe_get_class_id_postprocessor(template):
    if template.get_fixed_answer_choices_list():

        def postprocess_fn(output_or_target, example=None, is_target=False):
            output_or_target = strip_whitespace(output_or_target)
            return t5.data.postprocessors.string_label_to_class_id(
                output_or_target, label_classes=template.get_fixed_answer_choices_list()
            )

        return postprocess_fn

    else:
        return strip_whitespace


def get_tf_dataset(split, shuffle_files, seed, dataset_name, subset_name, template, split_mapping):
    # HF datasets does not support file-level shuffling
    del shuffle_files, seed
    dataset = datasets.load_dataset(dataset_name, subset_name)
    dataset = dataset[split_mapping[split]]
    dataset = utils.apply_template(dataset, template)
    return utils.hf_dataset_to_tf_dataset(dataset)

all_templates = promptsource.templates.TemplateCollection()
all_templates.remove("anli")

dataset_name = 'glue'
subset_name = 'cola'
task_name = 'mt5_test'
template_name = 'Make sense yes no'

dataset = all_templates.get_dataset(dataset_name, subset_name)
template = dataset[template_name]
# template = all_templates.get_dataset(dataset_name, subset_name)[template_name]

dataset_splits = utils.get_dataset_splits(dataset_name, subset_name)
# split_mapping = split_mapping or {k: k for k in dataset_splits.keys()}
split_mapping = None or {k: k for k in dataset_splits.keys()}
dataset_fn = functools.partial(
    get_tf_dataset,
    seed=None,
    dataset_name=dataset_name,
    subset_name=subset_name,
    template=template,
    split_mapping=split_mapping,
)

data_source = seqio.FunctionDataSource(
    dataset_fn,
    splits=list(split_mapping.keys()),
    num_input_examples={s: dataset_splits[split_mapping[s]].num_examples for s in split_mapping.keys()},
)
output_features = MT5_OUTPUT_FEATURES
preprocessors = [
    seqio.preprocessors.tokenize,
    seqio.preprocessors.append_eos,
    seqio.CacheDatasetPlaceholder(required=False),
]

# Add train and normal eval tasks
seqio.TaskRegistry.add(
    task_name,
    data_source,
    preprocessors=preprocessors,
    output_features=output_features,
    metric_fns=[GET_METRICS[m] for m in template.metadata.metrics],
    postprocess_fn=maybe_get_class_id_postprocessor(template),
)

