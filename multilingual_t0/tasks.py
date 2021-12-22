import re
import functools

from typing import Dict, List, Optional, Tuple

import seqio
import datasets
import tensorflow as tf

import t5
from t5.evaluation import metrics as mt
from t5.data.glue_utils import get_glue_metric, get_super_glue_metric

import promptsource.templates

from t5x.partitioning import LogicalAxisRules

seqio.add_global_cache_dirs(['gs://bigscience-t5x/multilingual_t0/v0.3'])

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

MAX_EXAMPLES_PER_DATASET = 500_000

DEFAULT_TEMPERATURE = 1.0 / 0.3
DEFAULT_MIX_RATE = functools.partial(
    t5.data.rate_num_examples, temperature=DEFAULT_TEMPERATURE)

MT5_SPM_PATH = "gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model"
MT5_VOCAB = t5.data.SentencePieceVocabulary(MT5_SPM_PATH)
MT5_OUTPUT_FEATURES = {
    "inputs": t5.data.Feature(
        vocabulary=MT5_VOCAB, add_eos=False, required=False),
    "targets": t5.data.Feature(
        vocabulary=MT5_VOCAB, add_eos=True)
}

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

def removeHyphen(example):
    example_clean = {}
    for key in example.keys():
        if "-" in key:
            new_key = key.replace("-", "_")
            example_clean[new_key] = example[key]
        else:
            example_clean[key] = example[key]
    example = example_clean
    return example


def feature_to_spec(feature, length=False):
    if isinstance(feature, datasets.ClassLabel):
        return tf.TensorSpec(shape=() if not length else (None if length == -1 else length,), dtype=tf.int64)
    elif isinstance(feature, datasets.Value):
        return tf.TensorSpec(
            shape=() if not length else (None if length == -1 else length,), dtype=getattr(tf.dtypes, feature.dtype)
        )
    elif hasattr(feature, "dtype") and hasattr(feature, "shape"):
        return tf.TensorSpec(shape=feature.shape, dtype=feature.dtype)
    elif isinstance(feature, datasets.Sequence):
        return feature_to_spec(feature.feature, length=feature.length)
    elif isinstance(feature, list):
        return [feature_to_spec(f, length=length) for f in feature]
    elif isinstance(feature, dict):
        return {k: feature_to_spec(v, length=length) for k, v in feature.items()}
    else:
        raise ValueError(f"Unparseable feature type {type(feature)}")


def hf_dataset_to_tf_dataset(dataset):
    return tf.data.Dataset.from_generator(
        dataset.__iter__, output_signature={k: feature_to_spec(v) for k, v in dataset.features.items()}
    )


def apply_template(dataset, template):
    def map_fn(ex):
        ex = removeHyphen(ex)
        inputs_and_targets = template.apply(ex)
        answer_choices = template.get_answer_choices_list(ex)
        if len(inputs_and_targets) == 2:
            inputs, targets = inputs_and_targets
            if targets == "":
                ex = {"inputs": inputs, "targets": "<NO LABEL>"}
            else:
                ex = {"inputs": inputs, "targets": targets}
        # When template results in an empty example, template.apply returns [""]
        # Also, if the template gets split wrong, len can be > 2
        # We will filter these out later
        else:
            ex = {"inputs": "", "targets": ""}

        if answer_choices:
            ex["answer_choices"] = answer_choices

        return ex

    def filter_fn(ex):
        return len(ex["inputs"]) > 0 and len(ex["targets"]) > 0

    original_columns = dataset.column_names
    dataset = dataset.map(map_fn).filter(filter_fn)
    # map keeps original columns, remove them
    return dataset.remove_columns(set(original_columns) - {"inputs", "targets", "answer_choices"})


def task_clean(text):
    # Clean the text according to allowed characters for a task name
    return re.sub(r"[^\w\d\._]+", "_", text)


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


def get_task_name(dataset_name, subset_name, template_name):
    return task_clean(dataset_name + (f"_{subset_name}_" if subset_name is not None else "_") + template_name)


def get_tf_dataset(split, shuffle_files, seed: Optional[int] = None, dataset_name=None, subset_name=None, template=None, split_mapping=None):
    # HF datasets does not support file-level shuffling
    del shuffle_files, seed
    dataset = datasets.load_dataset(dataset_name, subset_name)
    dataset = dataset[split_mapping[split]]
    dataset = apply_template(dataset, template)
    return hf_dataset_to_tf_dataset(dataset)


def add_task(dataset_name, subset_name=None, split_mapping=None):

    dataset = all_templates.get_dataset(dataset_name, subset_name)
    num_templates = len(dataset.all_template_names)

    info = datasets.get_dataset_infos(dataset_name)
    subset_info = subset_name or list(info.keys())[0]
    split_mapping = split_mapping or {k: k for k in info[subset_info].splits.keys()}
    dataset_splits = info[subset_info].splits
    train_size = dataset_splits['train'].num_examples

    if train_size > MAX_EXAMPLES_PER_DATASET:
        cap = MAX_EXAMPLES_PER_DATASET // num_templates
    else:
        cap = train_size

    if dataset_name == "glue":
        metrics = get_glue_metric(subset_name)
    elif dataset_name == "super_glue":
        if subset_name in ("wsc.fixed", "multirc"):
            # TODO: WSC and MultiRC need special pre/postprocesing
            metrics = [mt.accuracy]
        else:
            metrics = get_super_glue_metric(subset_name)
    else:
        # TODO what if metric is null?
        metrics = [] #[GET_METRICS[m] for m in dataset[dataset.all_template_names[0]].metadata.metrics]

    task_cap: Dict[str, int] = {}
    task_name_list = []
    for template_name in dataset.all_template_names:

        task_name = get_task_name(dataset_name, subset_name, template_name)
        task_cap[task_name] = cap
        task_name_list.append(task_name)

        template = dataset[template_name]

        dataset_fn = functools.partial(
            get_tf_dataset,
            dataset_name=dataset_name,
            subset_name=subset_name,
            template=template,
            split_mapping=split_mapping,
        )
        dataset_fn.__name__ = "dataset_fn"

        data_source = seqio.FunctionDataSource(
            dataset_fn=dataset_fn,
            splits=list(split_mapping.keys()),
            num_input_examples={s: dataset_splits[split_mapping[s]].num_examples for s in split_mapping.keys()},
        )

        # Add train and normal eval tasks
        seqio.TaskRegistry.add(
            task_name,
            source=data_source,
            preprocessors=[
                seqio.preprocessors.tokenize,
                seqio.preprocessors.append_eos,
                seqio.CacheDatasetPlaceholder(),
            ],
            output_features=MT5_OUTPUT_FEATURES,
            metric_fns=metrics,
            postprocess_fn=maybe_get_class_id_postprocessor(template),
        )

    return task_cap, task_name_list

# 3 stages of training/ablation: D4 -> GPT -> SuperGLUE
t0_train_mixture: List[str] = []
gpt_train_mixture: List[str] = []
sglue_train_mixture: List[str] = []
mixture_cap: Dict[str, int] = {}

all_templates = promptsource.templates.TemplateCollection()
all_templates.remove("anli")  # Need to special-case ANLI due to weird split conventions


# ==================================== OSCAR LM Adaptation ======================================
oscar_lm_adaptation_mixture: List[str] = []
OSCAR_LANGS = [
    'af', 'als', 'am', 'an', 'ar', 'arz', 'as', 'ast', 'av', 'az', 'azb', 'ba',
    'bar', 'bcl', 'be', 'bg', 'bh', 'bn', 'bo', 'bpy', 'br', 'bs', 'bxr', 'ca',
    'cbk', 'ce', 'ceb', 'ckb', 'cs', 'cv', 'cy', 'da', 'de', 'diq', 'dsb', 'dv', 
    'el', 'eml', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'frr', 'fy', 
    'ga', 'gd', 'gl', 'gn', 'gom', 'gu', 'he', 'hi', 'hr', 'hsb', 'ht', 'hu',
    'hy', 'ia', 'id', 'ie', 'ilo', 'io', 'is', 'it', 'ja', 'jbo', 'jv', 'ka',
    'kk', 'km', 'kn', 'ko', 'krc', 'ku', 'kv', 'kw', 'ky', 'la', 'lb', 'lez',
    'li', 'lmo', 'lo', 'lrc', 'lt', 'lv', 'mai', 'mg', 'mhr', 'min', 'mk', 'ml',
    'mn', 'mr', 'mrj', 'ms', 'mt', 'mwl', 'my', 'myv', 'mzn', 'nah', 'nap', 'nds',
    'ne', 'new', 'nl', 'nn', 'no', 'oc', 'or', 'os', 'pa', 'pam', 'pl', 'pms', 
    'pnb', 'ps', 'pt', 'qu', 'rm', 'ro', 'ru', 'sa', 'sah', 'scn', 'sd', 'sh',
    'si', 'sk', 'sl', 'so', 'sq', 'sr', 'su', 'sv', 'sw', 'ta', 'te', 'tg',
    'th', 'tk', 'tl', 'tr', 'tt', 'tyv', 'ug', 'uk', 'ur', 'uz', 'vec', 'vi',
    'vo', 'wa', 'war', 'wuu', 'xal', 'xmf', 'yi', 'yo', 'yue', 'zh'
]

def get_tf_dataset_oscar(split, shuffle_files, seed: Optional[int] = None, dataset_name=None, subset_name=None, split_mapping=None):

    def map_fn(ex):
        # return {"inputs": ex["text"], "targets": ex["text"]}
        return ex

    def filter_fn(ex):
        # return len(ex["inputs"]) > 0 and len(ex["targets"]) > 0
        return len(ex["text"]) > 0

    # HF datasets does not support file-level shuffling
    del shuffle_files, seed
    dataset = datasets.load_dataset(dataset_name, subset_name)
    dataset = dataset[split_mapping[split]]

    original_columns = dataset.column_names
    dataset = dataset.map(map_fn).filter(filter_fn)
    # map keeps original columns, remove them
    # dataset = dataset.remove_columns(set(original_columns) - {"inputs", "targets"})
    dataset = dataset.remove_columns(set(original_columns) - {"text"})
    return hf_dataset_to_tf_dataset(dataset)


info = datasets.get_dataset_infos("oscar")


for lang in OSCAR_LANGS:

    task_name = "oscar_{}_lm_objective".format(lang)
    subset_name = "unshuffled_deduplicated_{}".format(lang)
    split_mapping = {k: k for k in info[subset_name].splits.keys()}
    dataset_splits = info[subset_name].splits

    oscar_lm_adaptation_mixture.append(task_name)

    dataset_fn = functools.partial(
        get_tf_dataset_oscar,
        dataset_name="oscar",
        subset_name=subset_name,
        split_mapping=split_mapping,
    )
    dataset_fn.__name__ = "dataset_fn"

    data_source = seqio.FunctionDataSource(
        dataset_fn=dataset_fn,
        splits=list(split_mapping.keys()),
        num_input_examples={s: dataset_splits[split_mapping[s]].num_examples for s in split_mapping.keys()},
    )

    seqio.TaskRegistry.add(
        task_name,
        source=data_source,
        preprocessors=[
            functools.partial(
                seqio.preprocessors.rekey, key_map={
                    "inputs": None,
                    "targets": "text"
                }),
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            t5.data.preprocessors.targets_for_prefix_lm_objective,
            t5.data.preprocessors.pack_prefix_lm_encoder_decoder,
        ],
        output_features={
            "encoder_input_tokens": seqio.Feature(vocabulary=MT5_VOCAB, add_eos=False),
            "decoder_target_tokens": seqio.Feature(vocabulary=MT5_VOCAB, add_eos=False),
            "decoder_input_tokens": seqio.Feature(vocabulary=MT5_VOCAB, add_eos=False),
            "encoder_segment_ids": seqio.Feature(vocabulary=MT5_VOCAB, add_eos=False),
            "encoder_positions": seqio.Feature(vocabulary=MT5_VOCAB, add_eos=False),
            "decoder_segment_ids": seqio.Feature(vocabulary=MT5_VOCAB, add_eos=False),
            "decoder_positions": seqio.Feature(vocabulary=MT5_VOCAB, add_eos=False),
            "decoder_loss_weights": seqio.Feature(vocabulary=MT5_VOCAB, add_eos=False),
            # All but the last stage of the preprocessing uses "targets" as the key,
            # so this output feature is necessary. It is not marked required because
            # the final preprocessor drops it.
            "targets": seqio.Feature(vocabulary=MT5_VOCAB, add_eos=True),
        },
        metric_fns=[])

seqio.MixtureRegistry.add(
    "oscar_lm_adaptation",
    oscar_lm_adaptation_mixture,
    default_rate=DEFAULT_MIX_RATE,
)

seqio.MixtureRegistry.add(
    "oscar_no_en_lm_adaptation",
    [lang for lang in oscar_lm_adaptation_mixture if lang !="en"],
    default_rate=DEFAULT_MIX_RATE,
)


# ==================================== OPUS100 ======================================
opus100_lm_adaptation_mixture: List[str] = []
OPUS100_LANGS = [
    'af-en', 'am-en', 'an-en', 'ar-en', 'as-en', 'az-en', 'be-en', 'bg-en', 'bn-en', 'br-en', 'bs-en', 'ca-en',
    'cs-en', 'cy-en', 'da-en', 'de-en', 'dz-en', 'el-en', 'en-eo', 'en-es', 'en-et', 'en-eu', 'en-fa', 'en-fi',
    'en-fr', 'en-fy', 'en-ga', 'en-gd', 'en-gl', 'en-gu', 'en-ha', 'en-he', 'en-hi', 'en-hr', 'en-hu', 'en-hy',
    'en-id', 'en-ig', 'en-is', 'en-it', 'en-ja', 'en-ka', 'en-kk', 'en-km', 'en-ko', 'en-kn', 'en-ku', 'en-ky',
    'en-li', 'en-lt', 'en-lv', 'en-mg', 'en-mk', 'en-ml', 'en-mn', 'en-mr', 'en-ms', 'en-mt', 'en-my', 'en-nb',
    'en-ne', 'en-nl', 'en-nn', 'en-no', 'en-oc', 'en-or', 'en-pa', 'en-pl', 'en-ps', 'en-pt', 'en-ro', 'en-ru',
    'en-rw', 'en-se', 'en-sh', 'en-si', 'en-sk', 'en-sl', 'en-sq', 'en-sr', 'en-sv', 'en-ta', 'en-te', 'en-tg',
    'en-th', 'en-tk', 'en-tr', 'en-tt', 'en-ug', 'en-uk', 'en-ur', 'en-uz', 'en-vi', 'en-wa', 'en-xh', 'en-yi',
    'en-yo', 'en-zh', 'en-zu', 'ar-de', 'ar-fr', 'ar-nl', 'ar-ru', 'ar-zh', 'de-fr', 'de-nl', 'de-ru', 'de-zh',
    'fr-nl', 'fr-ru', 'fr-zh', 'nl-ru', 'nl-zh', 'ru-zh'
    ]

def get_tf_dataset_opus100(split, shuffle_files, seed: Optional[int] = None, dataset_name=None, subset_name=None, src_lang=None, tgt_lang=None, split_mapping=None):

    def map_fn(ex):
        return {"inputs": "Translate to {}: {}".format(tgt_lang, ex["translation"][src_lang]), "targets": ex["translation"][tgt_lang]}

    def filter_fn(ex):
        return len(ex["inputs"]) > 0 and len(ex["targets"]) > 0

    # HF datasets does not support file-level shuffling
    del shuffle_files, seed
    dataset = datasets.load_dataset(dataset_name, subset_name)
    dataset = dataset[split_mapping[split]]

    original_columns = dataset.column_names
    dataset = dataset.map(map_fn).filter(filter_fn)
    # map keeps original columns, remove them
    dataset = dataset.remove_columns(set(original_columns) - {"inputs", "targets", "answer_choices"})
    return hf_dataset_to_tf_dataset(dataset)

info = datasets.get_dataset_infos("opus100")
# subset_name = list(info.keys())[0]

for ori_lang in OPUS100_LANGS:

    lang_a, lang_b = ori_lang.split('-')
    split_mapping = {k: k for k in info[ori_lang].splits.keys()}
    dataset_splits = info[ori_lang].splits

    for src_lang, tgt_lang in [[lang_a, lang_b], [lang_b, lang_a]]:

        lang = "{}-{}".format(src_lang, tgt_lang)

        task_name = "opus100_{}_mt".format(lang.replace("-", "_"))
        opus100_lm_adaptation_mixture.append(task_name)

        dataset_fn = functools.partial(
            get_tf_dataset_opus100,
            dataset_name="opus100",
            subset_name=ori_lang,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            split_mapping=split_mapping,
        )
        dataset_fn.__name__ = "dataset_fn"

        data_source = seqio.FunctionDataSource(
            dataset_fn=dataset_fn,
            splits=list(split_mapping.keys()),
            num_input_examples={s: dataset_splits[split_mapping[s]].num_examples for s in split_mapping.keys()},
        )

        seqio.TaskRegistry.add(
            task_name,
            source=data_source,
            preprocessors=[
                seqio.preprocessors.tokenize,
                seqio.preprocessors.append_eos,
                seqio.CacheDatasetPlaceholder(),
            ],
            output_features=MT5_OUTPUT_FEATURES,
            metric_fns=[])

seqio.MixtureRegistry.add(
    "opus100_lm_adaptation",
    opus100_lm_adaptation_mixture,
    default_rate=DEFAULT_MIX_RATE,
)

# ==================================== Coreference Resolution ======================================
task_cap, task_name_list = add_task("super_glue", "wsc.fixed")
mixture_cap = {**mixture_cap, **task_cap}
# sglue_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("winogrande", "winogrande_xl")
mixture_cap = {**mixture_cap, **task_cap}

# ==================================== Natural Language Inference ======================================
task_cap, task_name_list = add_task("super_glue", "cb")
mixture_cap = {**mixture_cap, **task_cap}

task_cap, task_name_list = add_task("super_glue", "rte")
mixture_cap = {**mixture_cap, **task_cap}

# ANLI
# task_cap, task_name_list = add_task("dataset_name", "subset_name")
# ==================================== Paraphrase Identification ======================================
task_cap, task_name_list  = add_task("glue", "mrpc")
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

task_cap, task_name_list  = add_task("glue", "qqp")
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

task_cap, task_name_list  = add_task("paws", "labeled_final")
mixture_cap = {**mixture_cap, **task_cap}
#t0_train_mixture.extend(task_name_list)

# ==================================== Closed-Book QA ======================================
task_cap, task_name_list  = add_task("ai2_arc", "ARC-Challenge")
mixture_cap = {**mixture_cap, **task_cap}
gpt_train_mixture.extend(task_name_list)

task_cap, task_name_list  = add_task("ai2_arc", "ARC-Easy")
mixture_cap = {**mixture_cap, **task_cap}
gpt_train_mixture.extend(task_name_list)

task_cap, task_name_list  = add_task("kilt_tasks", "hotpotqa")
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

task_cap, task_name_list  = add_task("trivia_qa", "unfiltered")
mixture_cap = {**mixture_cap, **task_cap}
gpt_train_mixture.extend(task_name_list)

task_cap, task_name_list  = add_task("web_questions", None)
mixture_cap = {**mixture_cap, **task_cap}
gpt_train_mixture.extend(task_name_list)

task_cap, task_name_list  = add_task("wiki_qa", None)
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

# ==================================== Extractive QA ======================================
task_cap, task_name_list = add_task("adversarial_qa", "dbidaf")
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("adversarial_qa", "dbert")
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("adversarial_qa", "droberta")
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("duorc", "SelfRC")
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("duorc", "ParaphraseRC")
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("ropes", None)
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("squad_v2", None)
mixture_cap = {**mixture_cap, **task_cap}

task_cap, task_name_list = add_task("super_glue", "record")
mixture_cap = {**mixture_cap, **task_cap}
sglue_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("quoref", None)
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("tydiqa", None)
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

# ==================================== Multiple-Choice QA ======================================
task_cap, task_name_list = add_task("cos_e", "v1.11")
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("cosmos_qa", None)
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("dream", None)
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("openbookqa", "main")
mixture_cap = {**mixture_cap, **task_cap}

task_cap, task_name_list = add_task("qasc", None)
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("quail", None)
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("quarel", None)
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("quartz", None)
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("race", "high")
mixture_cap = {**mixture_cap, **task_cap}

task_cap, task_name_list = add_task("race", "middle")
mixture_cap = {**mixture_cap, **task_cap}

task_cap, task_name_list = add_task("sciq", None)
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("social_i_qa", None)
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("super_glue", "boolq")
mixture_cap = {**mixture_cap, **task_cap}
sglue_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("super_glue", "multirc")
mixture_cap = {**mixture_cap, **task_cap}
sglue_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("wiki_hop", "original")
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("wiqa", None)
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("piqa", None)
mixture_cap = {**mixture_cap, **task_cap}
gpt_train_mixture.extend(task_name_list)

# ==================================== Sentiment ======================================
task_cap, task_name_list = add_task("amazon_polarity", None)
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("app_reviews", None)
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("imdb", None)
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("rotten_tomatoes", None)
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("yelp_review_full", None)
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

# ==================================== Sentence Completion ======================================
task_cap, task_name_list = add_task("super_glue", "copa")
mixture_cap = {**mixture_cap, **task_cap}
# sglue_train_mixture.extend(task_name_list)

# task_cap, task_name_list = add_task("story_cloze", "2016")
# mixture_cap = {**mixture_cap, **task_cap}

task_cap, task_name_list = add_task("hellaswag", None)
mixture_cap = {**mixture_cap, **task_cap}
gpt_train_mixture.extend(task_name_list)
# ==================================== Structure-to-Text ======================================
task_cap, task_name_list = add_task("common_gen", None)
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("wiki_bio", None)
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

# ==================================== Summarization ======================================
task_cap, task_name_list = add_task("cnn_dailymail", "3.0.0")
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("gigaword", None)
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("multi_news", None)
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("samsum", None)
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("xsum", None)
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)
# ==================================== Topic Classification ======================================
task_cap, task_name_list = add_task("ag_news", None)
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("dbpedia_14", None)
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)

task_cap, task_name_list = add_task("trec", None)
mixture_cap = {**mixture_cap, **task_cap}
t0_train_mixture.extend(task_name_list)
# ==================================== Word Sense Disambiguation ======================================
task_cap, task_name_list = add_task("super_glue", "wic")
mixture_cap = {**mixture_cap, **task_cap}
sglue_train_mixture.extend(task_name_list)

TASK_BLACKLIST = [
    # Tasks which often tokenize to > 1024 tokens currently
    "hotpot_qa_distractor_Generate_Explanations",
    "hotpot_qa_fullwiki_Generate_Explanations",
    "hotpot_qa_distractor_Generate_Answer_and_Explanations",
    "hotpot_qa_fullwiki_Generate_Answer_and_Explanations",
    "hotpot_qa_fullwiki_Generate_Answer",
    "hotpot_qa_distractor_Generate_Answer",
    "hotpot_qa_distractor_Generate_Title_2",
    "hotpot_qa_fullwiki_Generate_Title_2",
    "hotpot_qa_fullwiki_Generate_Title_1",
    "hotpot_qa_distractor_Generate_Title_1",
    "hotpot_qa_distractor_Generate_Question",
    "hotpot_qa_fullwiki_Generate_Question",
    "tab_fact_tab_fact_tab_fact_3",
    "tab_fact_tab_fact_tab_fact_2",
    "tab_fact_tab_fact_tab_fact_1",
    "tab_fact_tab_fact_tab_fact_7",
    "tab_fact_tab_fact_tab_fact_4",
    "tab_fact_tab_fact_tab_fact_5",
    "tab_fact_tab_fact_tab_fact_6",
    "wiki_hop_masked_Choose_Best_Object_Candidate",
    "wiki_hop_masked_Indirect_Question_about_Birthplace_Citizenship_Place_of_Death",
    "narrativeqa_Template_05",
    "ecthr_cases_alleged_violation_prediction_silver_rationales",
    # Tasks with broken cached files
    "gigaword_summarize_",
]

seqio.MixtureRegistry.add(
    "t0_train",
    [task for task in t0_train_mixture if task not in TASK_BLACKLIST],
    default_rate=lambda t: mixture_cap[t.name],
)

seqio.MixtureRegistry.add(
    "t0_train_plus_opus",
    [task for task in t0_train_mixture+opus100_lm_adaptation_mixture if task not in TASK_BLACKLIST],
    default_rate=lambda t: mixture_cap[t.name],
)

seqio.MixtureRegistry.add(
    "gpt_train",
    [task for task in gpt_train_mixture if task not in TASK_BLACKLIST],
    default_rate=lambda t: mixture_cap[t.name],
)

seqio.MixtureRegistry.add(
    "sglue_train",
    [task for task in sglue_train_mixture if task not in TASK_BLACKLIST],
    default_rate=lambda t: mixture_cap[t.name],
)

seqio.MixtureRegistry.add(
    "t0_plus_train",
    [task for task in t0_train_mixture + gpt_train_mixture if task not in TASK_BLACKLIST],
    default_rate=lambda t: mixture_cap[t.name],
)

seqio.MixtureRegistry.add(
    "t0_plus_plus_train",
    [task for task in t0_train_mixture + gpt_train_mixture + sglue_train_mixture if task not in TASK_BLACKLIST],
    default_rate=lambda t: mixture_cap[t.name],
)


# ==================================== XCOPA ======================================
xcopa_eval_mixture: List[str] = []
LANGS = ['et', 'ht', 'it', 'id', 'qu', 'sw', 'zh', 'ta', 'th', 'tr', 'vi']

def get_tf_dataset_xcopa(split, shuffle_files, seed: Optional[int] = None, dataset_name=None, subset_name=None, split_mapping=None):

    def map_fn(ex):
        # return {"inputs": ex["text"], "targets": ex["text"]}
        return {
            "inputs": "Given the premise \"{premise}\" which is the most likely {question}? A: \"{choice1}\" or B: \"{choice2}\"".format(**ex),
            "targets": ["A", "B"][ex["label"]]
        }


    def filter_fn(ex):
        return len(ex["inputs"]) > 0 and len(ex["targets"]) > 0

    # HF datasets does not support file-level shuffling
    del shuffle_files, seed
    dataset = datasets.load_dataset(dataset_name, subset_name)
    dataset = dataset[split_mapping[split]]

    original_columns = dataset.column_names
    dataset = dataset.map(map_fn).filter(filter_fn)
    # map keeps original columns, remove them
    dataset = dataset.remove_columns(set(original_columns) - {"inputs", "targets"})
    return hf_dataset_to_tf_dataset(dataset)


info = datasets.get_dataset_infos("xcopa")


for lang in LANGS:

    task_name = "xcopa_{}".format(lang)
    subset_name = lang
    split_mapping = {k: k for k in info[subset_name].splits.keys()}
    dataset_splits = info[subset_name].splits

    xcopa_eval_mixture.append(task_name)

    dataset_fn = functools.partial(
        get_tf_dataset_xcopa,
        dataset_name="xcopa",
        subset_name=subset_name,
        split_mapping=split_mapping,
    )
    dataset_fn.__name__ = "dataset_fn"

    data_source = seqio.FunctionDataSource(
        dataset_fn=dataset_fn,
        splits=list(split_mapping.keys()),
        num_input_examples={s: dataset_splits[split_mapping[s]].num_examples for s in split_mapping.keys()},
    )

    seqio.TaskRegistry.add(
        task_name,
        source=data_source,
        preprocessors=[
            seqio.preprocessors.tokenize,
            seqio.preprocessors.append_eos,
            seqio.CacheDatasetPlaceholder(),
        ],
        output_features=MT5_OUTPUT_FEATURES,
        metric_fns=[mt.accuracy]
        )

seqio.MixtureRegistry.add(
    "xcopa_eval",
    xcopa_eval_mixture,
    default_rate=1.0
)


# ==================================== XNLI ======================================
xnli_eval_mixture: List[str] = []
LANGS = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]

def get_tf_dataset_xnli(split, shuffle_files, seed: Optional[int] = None, dataset_name=None, subset_name=None, split_mapping=None):

    def map_fn(ex):
        # return {"inputs": ex["text"], "targets": ex["text"]}
        return {
            "inputs": "{premise} Based on the previous passage, is it true that \"{hypothesis}\"? Yes, no, or maybe?".format(**ex),
            "targets": ["Yes", "Maybe", "No"][ex["label"]]
            # "inputs": "{premise} Based on the previous passage, is it true that \"{hypothesis}\"? True, False, or Possibly?".format(**ex),
            # "targets": ["True", "Possibly", "False"][ex["label"]]
        }

    def filter_fn(ex):
        return len(ex["inputs"]) > 0 and len(ex["targets"]) > 0

    # HF datasets does not support file-level shuffling
    del shuffle_files, seed
    dataset = datasets.load_dataset(dataset_name, subset_name)
    dataset = dataset[split_mapping[split]]

    original_columns = dataset.column_names
    dataset = dataset.map(map_fn).filter(filter_fn)
    # map keeps original columns, remove them
    dataset = dataset.remove_columns(set(original_columns) - {"inputs", "targets"})
    return hf_dataset_to_tf_dataset(dataset)


info = datasets.get_dataset_infos("xnli")


for lang in LANGS:

    task_name = "xnli_{}".format(lang)
    subset_name = lang
    split_mapping = {k: k for k in info[subset_name].splits.keys()}
    dataset_splits = info[subset_name].splits

    xnli_eval_mixture.append(task_name)

    dataset_fn = functools.partial(
        get_tf_dataset_xnli,
        dataset_name="xnli",
        subset_name=subset_name,
        split_mapping=split_mapping,
    )
    dataset_fn.__name__ = "dataset_fn"

    data_source = seqio.FunctionDataSource(
        dataset_fn=dataset_fn,
        splits=list(split_mapping.keys()),
        num_input_examples={s: dataset_splits[split_mapping[s]].num_examples for s in split_mapping.keys()},
    )

    seqio.TaskRegistry.add(
        task_name,
        source=data_source,
        preprocessors=[
            seqio.preprocessors.tokenize,
            seqio.preprocessors.append_eos,
            seqio.CacheDatasetPlaceholder(),
        ],
        postprocess_fn=t5.data.postprocessors.lower_text,
        output_features=MT5_OUTPUT_FEATURES,
        metric_fns=[mt.accuracy]
        )

seqio.MixtureRegistry.add(
    "xnli_eval",
    xnli_eval_mixture,
    default_rate=1.0
)

# ==================================== Flores ======================================
flores_mixture: List[str] = []
LANGS = [ "sien", "neen"]

def get_tf_dataset_flores(split, shuffle_files, seed: Optional[int] = None, dataset_name=None, subset_name=None, src_lang=None, tgt_lang=None, split_mapping=None):

    def map_fn(ex):
        return {"inputs": "Translate to {}: {}".format(tgt_lang, ex["translation"][src_lang]), "targets": ex["translation"][tgt_lang]}

    def filter_fn(ex):
        return len(ex["inputs"]) > 0 and len(ex["targets"]) > 0

    # HF datasets does not support file-level shuffling
    del shuffle_files, seed
    dataset = datasets.load_dataset(dataset_name, subset_name)
    dataset = dataset[split_mapping[split]]

    original_columns = dataset.column_names
    dataset = dataset.map(map_fn).filter(filter_fn)
    # map keeps original columns, remove them
    dataset = dataset.remove_columns(set(original_columns) - {"inputs", "targets", "answer_choices"})
    return hf_dataset_to_tf_dataset(dataset)

info = datasets.get_dataset_infos("flores")
# subset_name = list(info.keys())[0]

for lang in LANGS:

    split_mapping = {k: k for k in info[ori_lang].splits.keys()}
    dataset_splits = info[lang].splits
    lang_a, lang_b = lang[:2], lang[2:]

    for src_lang, tgt_lang in [[lang_a, lang_b], [lang_b, lang_a]]:

        task_name = "flores_{}_{}_mt".format(src_lang, tgt_lang)
        flores_mixture.append(task_name)

        dataset_fn = functools.partial(
            get_tf_dataset_flores,
            dataset_name="flores",
            subset_name=lang,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            split_mapping=split_mapping,
        )
        dataset_fn.__name__ = "dataset_fn"

        data_source = seqio.FunctionDataSource(
            dataset_fn=dataset_fn,
            splits=list(split_mapping.keys()),
            num_input_examples={s: dataset_splits[split_mapping[s]].num_examples for s in split_mapping.keys()},
        )

        seqio.TaskRegistry.add(
            task_name,
            source=data_source,
            preprocessors=[
                seqio.preprocessors.tokenize,
                seqio.preprocessors.append_eos,
                seqio.CacheDatasetPlaceholder(),
            ],
            output_features=MT5_OUTPUT_FEATURES,
            metric_fns=[mt.bleu]
            )

seqio.MixtureRegistry.add(
    "flores_mt_mixture",
    flores_mixture,
    default_rate=1.0
)

# seqio.MixtureRegistry.add(
#     "pawsx_eval",
#     xnli_eval_mixture
# )

# seqio.MixtureRegistry.add(
#     "xwino_eval",
#     xnli_eval_mixture
# )