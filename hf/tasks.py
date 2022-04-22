import os
import re
import functools
import multiprocessing

import numpy as np

from typing import Dict, List, Optional, Tuple, Any

import datasets
from datasets import Dataset, concatenate_datasets, interleave_datasets, get_dataset_infos, load_dataset, load_from_disk
from promptsource.templates import DatasetTemplates

from translation import add_translated_prompt_templates

MAX_EXAMPLES_PER_TASK = 500_000
DEFAULT_TEMPERATURE = 1.0 / 0.3
_num_proc = multiprocessing.cpu_count()

try:
    cache_path = os.environ["HF_DATASETS_CACHE"]

except Exception as e:
    print(e)
    cache_path = "~/.cache/mt0_tasks/"
    cache_path = os.path.expanduser(cache_path)

print("Using {}".format(cache_path))

# add_translated_prompt_templates()

class MixtureRegistry:
    """docstring for MixtureRegistry"""
    def __init__(self,
        mixture_name=None,
        mixture=[],
        temperature=DEFAULT_TEMPERATURE,
        max_examples=MAX_EXAMPLES_PER_TASK,
        include_translated=False,
        save_to_disk=cache_path,
        ):
        super(MixtureRegistry, self).__init__()

        self.mixture_name = mixture_name
        self.mixture = mixture
        self.temperature = temperature
        self.max_examples = max_examples
        self.include_translated = include_translated
        self.save_to_disk = save_to_disk

        # Dict to hold all task datasets and sampling probabilities
        self.task_dict = {}

        # def add_task(self, dataset_name=None, subset_name=None, mixture=None):
        for dataset_name, subset_name in self.mixture:

            if subset_name is not None:
                task_name = "{}_{}".format(dataset_name, subset_name)
            else:
                task_name = dataset_name

            dataset_templates = DatasetTemplates(
                dataset_name=dataset_name,
                subset_name=subset_name
                )

            dataset_samples = load_dataset(
                dataset_name,
                subset_name
            )

            if self.include_translated:
                template_list = dataset_templates.all_template_names
            else:
                #filter translated prompts
                template_list = [t for t in dataset_templates.all_template_names if "-translate-" not in t]

            num_templates = len(template_list)

            info = get_dataset_infos(dataset_name)
            subset_info = subset_name or list(info.keys())[0]
            dataset_splits = info[subset_info].splits

            train_size = dataset_splits['train'].num_examples

            if train_size*num_templates > self.max_examples:
                cap = self.max_examples // num_templates
            else:
                cap = train_size

            for idx, template in enumerate(template_list):
                template_save_path = os.path.join(
                    self.save_to_disk,
                    'tasks/{}/{}'.format(task_name, template)
                    )
                try:
                    dataset = load_from_disk(
                        template_save_path
                        )
                    print("Dataset+prompt already cached, loading from disk")
                except:
                    print("Dataset+prompt not yet cached")
                    try:
                        dataset_sub_samples = dataset_samples['train'].shuffle(seed=42+idx).select(range(0,cap))
                        dataset = self._apply_template(dataset_sub_samples, dataset_templates[template])
                        dataset.save_to_disk(template_save_path)
                        print("Cache Sucessful")

                    except Exception as e:
                        print(e)
                        print("Failed to cached this template")
                        print(dataset_templates[template].jinja)

                self.task_dict["{}_{}".format(task_name,template)] = {
                    'datasets': dataset,
                    'probabilities': cap,
                }

    def create_dataset(self):

        mixture_path =  os.path.join(
            self.save_to_disk,
            self.mixture_name
        )

        try:
            multitask_dataset = load_from_disk(mixture_path)
            return multitask_dataset
        except:
            task_dict = self.task_dict

            mixture = [task_dict[key]['datasets'] for key in task_dict]
            task_size = [task_dict[key]['probabilities'] for key in task_dict]
            task_sampling_rate = np.array(task_size)**self.temperature
            probabilities = task_sampling_rate/sum(task_sampling_rate)

            multitask_dataset = concatenate_datasets(
                mixture,
                )

            if self.save_to_disk != None:
                multitask_dataset.save_to_disk(
                    mixture_path
                    )

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

        if map_fn == None:
            map_fn = _map_fn

        original_columns = dataset.column_names
        dataset = dataset.map(
            map_fn,
            num_proc=_num_proc,
        ).filter(filter_fn)
        
        # map keeps original columns, remove them
        dataset = dataset.remove_columns(set(original_columns) - {"inputs", "labels", "answer_choices"})

        return dataset

t0_mixture = [
    ["glue", "mrpc"], #Paraphrase Identification
    ["glue", "qqp"],
    # ["paws", "labeled_final"],
    # ["kilt_tasks", "hotpotqa"], # Closed-Book QA
    # ["wiki_qa", None],
    # ["adversarial_qa", "dbidaf"], # Extractive QA
    # ["adversarial_qa", "dbert"],
    # ["adversarial_qa", "droberta"],
    # ["duorc", "SelfRC"],
    # ["duorc", "ParaphraseRC"],
    # # ["ropes", None],
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
    # # ["wiki_hop", "original"],
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

gpt_mixture = [
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

sglue_mixture = [
    ["super_glue", "wsc.fixed"], # Coreference Resolution
    ["super_glue", "record"], # Extractive QA
    ["super_glue", "boolq"], # Multiple-Choice QA
    ["super_glue", "multirc"], 
    ["super_glue", "copa"], # Sentence Completion
    ["super_glue", "wic"], # Word Sense Disambiguation
]

training_mixtures = {
    "t0_train": MixtureRegistry(
        mixture_name="t0_train",
        mixture=t0_mixture
        ).create_dataset(),
    # "t0_plus_train": MixtureRegistry(
    #     mixture_name="t0_plus_train",
    #     mixture=t0_mixture+gpt_mixture
    #     ).create_dataset(),
    # "t0_plus_plus_train": MixtureRegistry(
    #     mixture_name="t0_plus_plus_train",
    #     mixture=t0_mixture+gpt_mixture+sglue_mixture
    #     ).create_dataset(),
    # "translated_t0_train": MixtureRegistry(
    #     mixture_name="translated_t0_train",
    #     mixture=t0_mixture,
    #     include_translated=True
    #     ).create_dataset(),
    # "translated_t0_plus_train": MixtureRegistry(
    #     mixture_name="translated_t0_plus_train",
    #     mixture=t0_mixture+gpt_mixture,
    #     include_translated=True
    #     ).create_dataset(),
    # "translated_t0_plus_plus_train": MixtureRegistry(
    #     mixture_name="translated_t0_plus_plus_train",
    #     mixture=t0_mixture+gpt_mixture+sglue_mixture,
    #     include_translated=True
    #     ).create_dataset(),
}
