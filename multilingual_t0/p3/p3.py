# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
# TODO: Address all TODOs and remove all explanatory comments
"""TODO: Add a description here."""


import csv
import json
import os

import datasets

import seqio

_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2022}
}
"""

_DESCRIPTION = "Seqio cached dataset for mT0"
_HOMEPAGE = ""
_LICENSE = ""
_URLS = {}


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class P3(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    # VERSION = datasets.Version("0.0.1")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=task,
            # version=VERSION,
            description=_DESCRIPTION) \
        for task in list(seqio.MixtureRegistry.names())
    ]

    DEFAULT_CONFIG_NAME = "mt0_train_mixture"

    def _info(self):

        features = datasets.Features(
            {
                "inputs"                : datasets.Sequence(datasets.Value("int32")),
                "targets"               : datasets.Sequence(datasets.Value("int32")),
                # "answer_choices"        : datasets.Value("string"),
                # "inputs_pretokenized"   : datasets.Value("string"),
                # "targets_pretokenized"  : datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )


    def _split_generators(self, dl_manager):

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test",
                },
            ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.VALIDATION,
            #     gen_kwargs={
            #         "split": "validation",
            #     },
            # ),
        ]


    def _generate_examples(self, split):

        raw_dataset = seqio.get_mixture_or_task(self.config.name).get_dataset(
            sequence_length={"inputs": 1024, "targets": 256},
            split=split,
            num_epochs=1,
            use_cached=True,
            seed=42
        )

        for idx, batch in enumerate(raw_dataset.as_numpy_iterator()):
            yield idx, {
                "inputs"                : batch['inputs'],
                "targets"               : batch['targets'],
                # "answer_choices"        : list(batch['answer_choices'].numpy().astype(str)),
                # "inputs_pretokenized"   : batch["inputs_pretokenized"].numpy().decode('UTF-8'),
                # "targets_pretokenized"  : batch["targets_pretokenized"].numpy().decode('UTF-8'),
                }