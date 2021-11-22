import re
import csv
import seqio
import functools

from typing import Dict, List, Optional, Tuple

import datasets
import pkg_resources
import tensorflow as tf
import tensorflow_datasets as tfds

import t5
from t5.evaluation import metrics as mt
from t5.data.glue_utils import get_glue_metric, get_super_glue_metric

import promptsource.templates

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


def get_dataset_splits(dataset_name, subset_name=None):
    info = datasets.get_dataset_infos(dataset_name)
    subset_name = subset_name or list(info.keys())[0]
    return info[subset_name].splits


def task_clean(text):
    # Clean the text according to allowed characters for a task name
    return re.sub(r"[^\w\d\._]+", "_", text)


def get_task_name(dataset_name, subset_name, template_name):
    return task_clean(dataset_name + (f"_{subset_name}_" if subset_name is not None else "_") + template_name)

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


def get_tf_dataset(split, shuffle_files, seed: Optional[int] = None, dataset_name=None, subset_name=None, template=None, split_mapping=None):
    # HF datasets does not support file-level shuffling
    del shuffle_files, seed
    dataset = datasets.load_dataset(dataset_name, subset_name)
    dataset = dataset[split_mapping[split]]
    dataset = apply_template(dataset, template)
    return hf_dataset_to_tf_dataset(dataset)


def add_task(dataset_name, subset_name, template_name, task_name=None, split_mapping=None):
    template = all_templates.get_dataset(dataset_name, subset_name)[template_name]
    task_name = task_name or get_task_name(dataset_name, subset_name, template_name)

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
        metrics = [GET_METRICS[m] for m in template.metadata.metrics]

    dataset_splits = get_dataset_splits(dataset_name, subset_name)
    split_mapping = split_mapping or {k: k for k in dataset_splits.keys()}

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

    preprocessors = [
        seqio.preprocessors.tokenize,
        seqio.preprocessors.append_eos,
        seqio.CacheDatasetPlaceholder(),
    ]

    # Add train and normal eval tasks
    seqio.TaskRegistry.add(
        task_name,
        source=data_source,
        preprocessors=preprocessors,
        output_features=MT5_OUTPUT_FEATURES,
        metric_fns=metrics,
        postprocess_fn=maybe_get_class_id_postprocessor(template),
    )

    # # Add rank classification eval task
    # if template.answer_choices:
    #     rank_classification_preprocessor = functools.partial(
    #         t5.data.preprocessors.rank_classification,
    #         inputs_fn=lambda ex: tf.fill((len(ex["answer_choices"]),), ex["inputs"]),
    #         targets_fn=lambda ex: ex["answer_choices"],
    #         is_correct_fn=lambda ex: tf.equal(ex["answer_choices"], tf.strings.strip(ex["targets"])),
    #         weight_fn=lambda ex: 1.0,
    #     )

    #     fixed_choices = template.get_fixed_answer_choices_list()
    #     num_classes = len(fixed_choices) if fixed_choices else None
    #     seqio.TaskRegistry.add(
    #         task_name + "_score_eval",
    #         source=data_source,
    #         preprocessors=[rank_classification_preprocessor] + preprocessors,
    #         output_features=output_features,
    #         metric_fns=[functools.partial(t5.evaluation.metrics.rank_classification, num_classes=num_classes)],
    #         postprocess_fn=t5.data.postprocessors.rank_classification,
    #     )

datatset_subset_tuple = Tuple[str, Optional[str]]
d4_train: List[datatset_subset_tuple] = []
d4_eval: List[datatset_subset_tuple] = []
d3_train_gpt: List[datatset_subset_tuple] = []
d3_train_sglue: List[datatset_subset_tuple] = []
bias_fairness_eval: List[datatset_subset_tuple] = []
gsheet: Dict[datatset_subset_tuple, Dict] = {}
experiment_path = pkg_resources.resource_filename(__name__, "experiment_D4.csv")
with open(experiment_path) as exp_file:
    reader = csv.DictReader(exp_file)
    for row in reader:
        if row["skip"]:
            continue
        if row["subset"] == "":
            row["subset"] = None  # to match promptsource.Template object
        dataset_subset = (row["HF_name"], row["subset"])
        if row["do_train"] == "TRUE":
            d4_train.append(dataset_subset)
        if row["do_eval"] == "TRUE":
            d4_eval.append(dataset_subset)
        if row["D3_do_train"] == "TRUE" and "GPT" in row["seed_paper"]:
            d3_train_gpt.append(dataset_subset)
        if row["D3_do_train"] == "TRUE" and row["HF_name"] == "super_glue":
            d3_train_sglue.append(dataset_subset)
        if (
            row["do_eval"] == "TRUE"
            and row["task_by_convention"] == "bias_and_fairness"
            and row["HF_name"] != "winogender"
        ):
            bias_fairness_eval.append(dataset_subset)
        gsheet[dataset_subset] = row
all_datasets = d4_train + d4_eval + d3_train_gpt + d3_train_sglue + bias_fairness_eval

all_templates = promptsource.templates.TemplateCollection()
all_templates.remove("anli")  # Need to special-case ANLI due to weird split conventions

# winogrande_winogrande_xl_underscore_refer_to
add_task('winogrande', 'winogrande_xl', 'does underscore refer to')
print(seqio.TaskRegistry.names())

# # 3 stages of training/ablation: D4 -> GPT -> SuperGLUE
# d4_train_mixture: List[str] = []  # strings are dataset_subset_template
# gpt_train_mixture: List[str] = []
# sglue_train_mixture: List[str] = []
# d4_eval_mixture: List[str] = []
# bias_fairness_eval_mixture: List[str] = []
# mixture_cap: Dict[str, int] = {}
# single_original_task: Dict[Tuple[str, str], str] = {}
# all_original_tasks: List[str] = []
# for dataset_name, subset_name in all_templates.keys:
#     if (dataset_name, subset_name) not in all_datasets:
#         all_templates.remove(dataset_name, subset_name)
#         continue

#     dataset = all_templates.get_dataset(dataset_name, subset_name)
#     num_templates = len(dataset.all_template_names)
#     train_size = gsheet[(dataset_name, subset_name)]["train_size"]
#     if train_size == "":
#         train_size = 0
#     else:
#         train_size = int(train_size)
#     if train_size > MAX_EXAMPLES_PER_DATASET:
#         cap = MAX_EXAMPLES_PER_DATASET // num_templates
#     else:
#         cap = train_size
#     for template_name in dataset.all_template_names:
#         add_task(dataset_name, subset_name, template_name)

#         template = dataset[template_name]

#         task_name = get_task_name(dataset_name, subset_name, template_name)

#         if (dataset_name, subset_name) not in single_original_task and template.metadata.original_task:
#             single_original_task[(dataset_name, subset_name)] = task_name

#         if template.metadata.original_task:
#             all_original_tasks.append(task_name)

#         if (dataset_name, subset_name) in d4_train:
#             d4_train_mixture.append(task_name)
#             mixture_cap[task_name] = cap
#         if (dataset_name, subset_name) in d3_train_gpt:
#             gpt_train_mixture.append(task_name)
#             mixture_cap[task_name] = cap
#         if (dataset_name, subset_name) in d3_train_sglue:
#             sglue_train_mixture.append(task_name)
#             mixture_cap[task_name] = cap
#         if (dataset_name, subset_name) in d4_eval:
#             if template.metadata.original_task:
#                 d4_eval_mixture.append(task_name)
#             # TODO use template.metadata.answer_choices here for rank eval
#         if (dataset_name, subset_name) in bias_fairness_eval:
#             bias_fairness_eval_mixture.append(task_name)

# # Special case for ANLI, which has weirdly-named splits and rounds that should be subsets
# dataset_name, subset_name = ("anli", None)
# dataset = all_templates.get_dataset(dataset_name, subset_name)
# for anli_round in ("r1", "r2", "r3"):
#     for template_name in all_templates.get_dataset(dataset_name, subset_name).all_template_names:
#         task_name = get_task_name(dataset_name, subset_name, template_name) + f"_{anli_round}"
#         split_mapping = {
#             "train": f"train_{anli_round}",
#             "validation": f"dev_{anli_round}",
#             "test": f"test_{anli_round}",
#         }
#         add_task(dataset_name, subset_name, template_name, task_name, split_mapping)

#         template = dataset[template_name]
#         if template.metadata.original_task:
#             d4_eval_mixture.append(task_name)  # TODO or add to ANLI special mixture
#         # TODO use template.metadata.answer_choices here for rank eval


# TASK_BLACKLIST = [
#     # Tasks which often tokenize to > 1024 tokens currently
#     "hotpot_qa_distractor_Generate_Explanations",
#     "hotpot_qa_fullwiki_Generate_Explanations",
#     "hotpot_qa_distractor_Generate_Answer_and_Explanations",
#     "hotpot_qa_fullwiki_Generate_Answer_and_Explanations",
#     "hotpot_qa_fullwiki_Generate_Answer",
#     "hotpot_qa_distractor_Generate_Answer",
#     "hotpot_qa_distractor_Generate_Title_2",
#     "hotpot_qa_fullwiki_Generate_Title_2",
#     "hotpot_qa_fullwiki_Generate_Title_1",
#     "hotpot_qa_distractor_Generate_Title_1",
#     "hotpot_qa_distractor_Generate_Question",
#     "hotpot_qa_fullwiki_Generate_Question",
#     "tab_fact_tab_fact_tab_fact_3",
#     "tab_fact_tab_fact_tab_fact_2",
#     "tab_fact_tab_fact_tab_fact_1",
#     "tab_fact_tab_fact_tab_fact_7",
#     "tab_fact_tab_fact_tab_fact_4",
#     "tab_fact_tab_fact_tab_fact_5",
#     "tab_fact_tab_fact_tab_fact_6",
#     "wiki_hop_masked_Choose_Best_Object_Candidate",
#     "wiki_hop_masked_Indirect_Question_about_Birthplace_Citizenship_Place_of_Death",
#     "narrativeqa_Template_05",
#     "ecthr_cases_alleged_violation_prediction_silver_rationales",
#     # Tasks with broken cached files
#     "gigaword_summarize_",
# ]

# # Tasks that failed caching (won't try to fix them for now) - remove when we are done
# D4_TRAIN_SCORE_EVAL_TASK_BLACKLIST = [
#     "amazon_polarity_Is_this_product_review_positive_score_eval",
#     "amazon_polarity_Is_this_review_negative_score_eval",
#     "amazon_polarity_Is_this_review_score_eval",
#     "amazon_polarity_User_recommend_this_product_score_eval",
#     "amazon_polarity_convey_negative_or_positive_sentiment_score_eval",
#     "amazon_polarity_flattering_or_not_score_eval",
#     "amazon_polarity_negative_or_positive_tone_score_eval",
#     "amazon_polarity_user_satisfied_score_eval",
#     "amazon_polarity_would_you_buy_score_eval",
#     "dbpedia_14_given_a_choice_of_categories__score_eval",
#     "dbpedia_14_given_list_what_category_does_the_paragraph_belong_to_score_eval",
#     "dbpedia_14_pick_one_category_for_the_following_text_score_eval",
#     "wiki_hop_original_choose_best_object_affirmative_1_score_eval",
#     "wiki_hop_original_choose_best_object_affirmative_2_score_eval",
#     "wiki_hop_original_choose_best_object_affirmative_3_score_eval",
#     "wiki_hop_original_choose_best_object_interrogative_1_score_eval",
#     "wiki_hop_original_choose_best_object_interrogative_2_score_eval",
# ]

# seqio.MixtureRegistry.add(
#     "d4_train",
#     [task for task in d4_train_mixture if task not in TASK_BLACKLIST],
#     default_rate=lambda t: mixture_cap[t.name],
# )

# seqio.MixtureRegistry.add(
#     "gpt_train",
#     [task for task in gpt_train_mixture if task not in TASK_BLACKLIST],
#     default_rate=lambda t: mixture_cap[t.name],
# )

# seqio.MixtureRegistry.add(
#     "sglue_train",
#     [task for task in sglue_train_mixture if task not in TASK_BLACKLIST],
#     default_rate=lambda t: mixture_cap[t.name],
# )

# seqio.MixtureRegistry.add(
#     "d4_gpt_train",
#     [task for task in d4_train_mixture + gpt_train_mixture if task not in TASK_BLACKLIST],
#     default_rate=lambda t: mixture_cap[t.name],
# )

# seqio.MixtureRegistry.add(
#     "d4_gpt_sglue_train",
#     [task for task in d4_train_mixture + gpt_train_mixture + sglue_train_mixture if task not in TASK_BLACKLIST],
#     default_rate=lambda t: mixture_cap[t.name],
# )

# seqio.MixtureRegistry.add(
#     "d4_eval",
#     [task for task in d4_eval_mixture if task not in TASK_BLACKLIST],
#     default_rate=functools.partial(seqio.mixing_rate_num_examples, maximum=500_000),
# )  # eval mixture does not need to be capped


# # seqio.MixtureRegistry.add(
# #     "d4_score_eval",
# #     [
# #         task
# #         for task in seqio.TaskRegistry.names()
# #         if task.endswith("_score_eval")
# #         and task.split("_score_eval")[0] in d4_eval_mixture
# #         and task.split("_score_eval")[0] not in TASK_BLACKLIST
# #     ],
# #     default_rate=functools.partial(seqio.mixing_rate_num_examples, maximum=500_000),
# # )

# # # Train tasks we don't care about evaluating on
# # D4_TRAIN_SKIP_EVAL = [
# #     "paws_labeled_final",
# #     "adversarial_qa_dbidaf",
# #     "adversarial_qa_dbert",
# #     "duorc_ParaphraseRC",
# #     "dream",
# #     "amazon_polarity",
# #     "app_reviews",
# #     "imdb",
# #     "wiki_bio",
# #     "gigaword",
# #     "multi_news",
# #     "samsum",
# #     "dbpedia_14",
# #     "trec",
# # ]

# # seqio.MixtureRegistry.add(
# #     "d4_train_eval",
# #     [
# #         task
# #         for task in d4_train_mixture
# #         if task not in TASK_BLACKLIST
# #         and not any([skip in task for skip in D4_TRAIN_SKIP_EVAL])
# #         and task in all_original_tasks
# #     ],
# #     default_rate=lambda t: mixture_cap[t.name],
# # )

# # seqio.MixtureRegistry.add(
# #     "d4_train_score_eval",
# #     [
# #         task
# #         for task in seqio.TaskRegistry.names()
# #         if task.endswith("_score_eval")
# #         and task.split("_score_eval")[0] in d4_train_mixture
# #         and task.split("_score_eval")[0] not in TASK_BLACKLIST
# #         and task not in D4_TRAIN_SCORE_EVAL_TASK_BLACKLIST
# #         and not any([skip in task for skip in D4_TRAIN_SKIP_EVAL])
# #         and task.split("_score_eval")[0] in all_original_tasks
# #     ],
# #     default_rate=functools.partial(seqio.mixing_rate_num_examples, maximum=500_000),
# # )

# # seqio.MixtureRegistry.add(
# #     "d4_train_one_og_prompt",
# #     [task for task in single_original_task.values() if task in d4_train_mixture and task not in TASK_BLACKLIST],
# #     default_rate=lambda t: mixture_cap[t.name],
# # )

# # seqio.MixtureRegistry.add(
# #     "d4_train_all_og_prompts",
# #     [task for task in all_original_tasks if task in d4_train_mixture and task not in TASK_BLACKLIST],
# #     default_rate=lambda t: mixture_cap[t.name],
# # )

# # seqio.MixtureRegistry.add(
# #     "bias_fairness_eval",
# #     bias_fairness_eval_mixture,
# #     default_rate=functools.partial(seqio.mixing_rate_num_examples, maximum=500_000),
# # )

# # seqio.MixtureRegistry.add(
# #     "bias_fairness_eval_score_eval",
# #     [
# #         task
# #         for task in seqio.TaskRegistry.names()
# #         if task.endswith("_score_eval") and task.split("_score_eval")[0] in bias_fairness_eval_mixture
# #     ],
# #     default_rate=functools.partial(seqio.mixing_rate_num_examples, maximum=500_000),
# # )
