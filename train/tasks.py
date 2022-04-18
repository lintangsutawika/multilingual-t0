import re
import functools

import numpy as np
import multiprocessing

from typing import Dict, List, Optional, Tuple, Any

import datasets
from datasets import Dataset, concatenate_datasets, interleave_datasets, get_dataset_infos, load_dataset, load_from_disk
from promptsource.templates import DatasetTemplates

from translation import add_translated_prompt_templates

MAX_EXAMPLES_PER_DATASET = 500_000
DEFAULT_TEMPERATURE = 1.0 / 0.3
_num_proc = multiprocessing.cpu_count()

# add_translated_prompt_templates()

class MixtureRegistry:
    """docstring for MixtureRegistry"""
    def __init__(self,
        temperature=DEFAULT_TEMPERATURE,
        max_examples=MAX_EXAMPLES_PER_DATASET,
        include_translated=False,
        save_to_disk=None, #"~/.cache",
        mixture=None,
        ):
        super(MixtureRegistry, self).__init__()
        self.temperature = temperature
        self.max_examples = max_examples
        self.include_translated = include_translated
        self.save_to_disk = save_to_disk

        # Dict to hold all task datasets and sampling probabilities
        self.task_dict = {}

        if mixture is not None:
            self.add_task(mixture=mixture)


    def add_task(self, dataset_name=None, subset_name=None, mixture=None):


        if dataset_name is not None:

            dataset_templates = DatasetTemplates(
                dataset_name=dataset_name,
                subset_name=subset_name
                )

            if subset_name is not None:
                task_name = "{}_{}".format(dataset_name, subset_name)
            else:
                task_name = dataset_name

            if self.include_translated:
                template_list = dataset_templates.all_template_names
            else:
                #filter translated prompts
                template_list = [t for t in dataset_templates.all_template_names if "-translate-" not in t]

            num_templates = len(template_list)

            info = get_dataset_infos(dataset_name)
            subset_info = subset_name or list(info.keys())[0]
            dataset_splits = info[subset_info].splits

            if 'train' in dataset_splits:
                train_size = dataset_splits['train'].num_examples

                if train_size*num_templates > MAX_EXAMPLES_PER_DATASET:
                    cap = MAX_EXAMPLES_PER_DATASET // num_templates
                else:
                    cap = train_size

                print("Here:", dataset_name, subset_name)
                task_dataset = load_dataset(
                    dataset_name,
                    subset_name,
                )

                for template in template_list:
                    try:
                        dataset = task_dataset['train'].shuffle(seed=42).select(range(0,cap))
                        self.task_dict["{}_{}".format(task_name,template)] = {
                            'datasets': self._apply_template(dataset, dataset_templates[template]),
                            'probabilities': cap,
                        }
                    except:
                        print("Failed to cached this template")
                        print(dataset_templates[template].jinja)

        elif mixture is not None:
            for mix in mixture:
                self.task_dict = {**self.task_dict, **mix.task_dict}


    def create_dataset(self, task_dict=None):

        if task_dict is None:
            task_dict = self.task_dict

        task_list = [task_dict[key]['datasets'] for key in task_dict]
        task_size = [task_dict[key]['probabilities'] for key in task_dict]
        task_sampling_rate = np.array(task_size)**self.temperature
        probabilities = task_sampling_rate/sum(task_sampling_rate)

        # multitask_dataset = interleave_datasets(
        multitask_dataset = concatenate_datasets(
            task_list,
            # probabilities=probabilities,
            # seed=42
            )

        if self.save_to_disk != None:
            multitask_dataset.save_to_disk(self.save_to_disk)

        return multitask_dataset


    def _apply_template(self, dataset, template, map_fn=None):
        def _map_fn(ex):

            inputs_and_targets = template.apply(ex)
            answer_choices = template.get_answer_choices_list(ex)

            if len(inputs_and_targets) == 2:
                inputs, targets = inputs_and_targets
                if targets == "":
                    ex = {"inputs": inputs, "labels": "<NO LABEL>"}
                else:
                    ex = {"inputs": inputs, "labels": targets}

            else:
                ex = {"inputs": "", "labels": ""}

            if answer_choices:
                ex["answer_choices"] = answer_choices

            return ex

        def filter_fn(ex):
            return len(ex["inputs"]) > 0 and len(ex["labels"]) > 0

        if self.save_to_disk is not None:
            try:
                dataset = load_from_disk(self.save_to_disk)
                print("Dataset+prompt already cached, loading from disk")
                return dataset
            except:
                print("Dataset+prompt not yet cached")

        if map_fn == None:
            map_fn = _map_fn

        original_columns = dataset.column_names
        dataset = dataset.map(
            map_fn,
            # num_proc=_num_proc,
        ).filter(filter_fn)
        
        # map keeps original columns, remove them
        dataset = dataset.remove_columns(set(original_columns) - {"inputs", "labels", "answer_choices"})

        if self.save_to_disk is not None:
            dataset.save_to_disk(self.save_to_disk)

        return dataset


class CustomTemplate(object):
    """docstring for CustomTemplate"""
    def __init__(self, inputs_fn, targets_fn):
        super(CustomTemplate, self).__init__()
        self.inputs_fn = inputs_fn
        self.targets_fn = targets_fn

    def get_answer_choices_list(self, example):
        return None

    def apply(self, example, truncate=True, highlight_variables=False):
        inputs = self.inputs_fn(example)
        targets = self.targets_fn(example)
        return inputs, targets

t0_task_list = [
    # ["wiki_hop", "original"], # (# Multiple-Choice QA)
    ["glue", "mrpc"], #Paraphrase Identification
    # ["glue", "qqp"],
    # ["paws", "labeled_final"],
    ["kilt_tasks", "hotpotqa"], # Closed-Book QA
    # ["wiki_qa", None],
    # ["adversarial_qa", "dbidaf"], # Extractive QA
    # ["adversarial_qa", "dbert"],
    # ["adversarial_qa", "droberta"],
    # ["duorc", "SelfRC"],
    # ["duorc", "ParaphraseRC"],
    # ["ropes", None],
    # ["quoref", None],
    # ["cos_e", "v1.11"], # Multiple-Choice QA
    # ["cosmos_qa", None],
    # ["dream", None],
    # ["qasc", None],
    # ["quail", None],
    # ["quarel", None],
    # ["quartz", None],
    # ["sciq", None],
    # ["social_i_qa", None],
    # ["wiqa", None],
    # ["amazon_polarity", None], # Sentiment
    # ["app_reviews", None],
    # ["imdb", None],
    # ["rotten_tomatoes", None],
    # ["yelp_review_full", None],
    # ["common_gen", None], # Structure-to-Text
    # ["wiki_bio", None],
    # ["cnn_dailymail", "3.0.0"], # Summarization
    # ["gigaword", None],
    # ["multi_news", None],
    # ["samsum", None],
    # ["xsum", None],
    # ["ag_news", None], # Topic Classification
    # ["dbpedia_14", None],
    # ["trec", None], 
]

gpt_task_list = [
    ["ai2_arc", "ARC-Challenge"], #Closed-Book QA
    ["ai2_arc", "ARC-Easy"],
    ["trivia_qa", "unfiltered"],
    ["web_questions", None],
    ["openbookqa", "main"], # Multiple-Choice QA
    ["race", "high"],
    ["race", "middle"],
    ["piqa", None],
    ["hellaswag", None], # Sentence Completion
]

sglue_task_list = [
    ["super_glue", "wsc.fixed"], # Coreference Resolution
    ["super_glue", "record"], # Extractive QA
    ["super_glue", "boolq"], # Multiple-Choice QA
    ["super_glue", "multirc"], 
    ["super_glue", "copa"], # Sentence Completion
    ["super_glue", "wic"], # Word Sense Disambiguation
]

# 3 stages of training/ablation: D4 -> GPT -> SuperGLUE
t0_train_mixture = MixtureRegistry()
for task in t0_task_list:
    t0_train_mixture.add_task(*task)

training_mixtures = {
    "t0_train": MixtureRegistry(
        mixture=[
            t0_train_mixture
            ]
        ),
}




# gpt_train_mixture = MixtureRegistry()
# for task in gpt_task_list:
#     gpt_train_mixture.add_task(*task)

# sglue_train_mixture = MixtureRegistry()
# for task in sglue_task_list:
#     sglue_train_mixture.add_task(*task)

# translated_t0_train_mixture = MixtureRegistry(include_translated=True)
# for task in t0_task_list:
#     translated_t0_train_mixture.add_task(*task)

# translated_gpt_train_mixture = MixtureRegistry(include_translated=True)
# for task in gpt_task_list:
#     translated_gpt_train_mixture.add_task(*task)

# translated_sglue_train_mixture = MixtureRegistry(include_translated=True)
# for task in sglue_task_list:
#     translated_sglue_train_mixture.add_task(*task)

# training_mixtures = {
#     "t0_train": MixtureRegistry(
#         mixture=[
#             t0_train_mixture
#             ]
#         ),
#     "t0_plus_train": MixtureRegistry(
#         mixture=[
#             t0_train_mixture,
#             gpt_train_mixture
#             ]
#         ),
#     "t0_plus_plus_train": MixtureRegistry(
#         mixture=[
#             t0_train_mixture,
#             gpt_train_mixture,
#             sglue_train_mixture
#             ]
#         ),
#     "translated_t0_train": MixtureRegistry(
#         mixture=[
#             translated_t0_train_mixture
#             ]
#         ),
#     "translated_t0_plus_train": MixtureRegistry(
#         mixture=[
#             translated_t0_train_mixture,
#             translated_gpt_train_mixture
#             ]
#         ),
#     "translated_t0_plus_plus_train": MixtureRegistry(
#         mixture=[
#             translated_t0_train_mixture,
#             translated_gpt_train_mixture,
#             translated_sglue_train_mixture
#             ]
#         ),
# }


