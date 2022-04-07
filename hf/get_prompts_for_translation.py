import json

import numpy as np
import pandas as pd

from promptsource.templates import DatasetTemplates

dataset_lists_df = pd.read_csv('datasets.csv')
dataset_lists_df = dataset_lists_df[dataset_lists_df['do_train'].notnull()]

template_string_list = []
template_name_list = []

template_entities = {}
for row in dataset_lists_df.iterrows():
    idx, row_items = row
    dataset_name = row_items['HF_name']
    subset_name = None if pd.isna(row_items['subset']) else row_items['subset']

    dataset_templates = DatasetTemplates(
        dataset_name=dataset_name,
        subset_name=subset_name
        )

    num_templates = len(dataset_templates)
    template_list = dataset_templates.all_template_names

    for template in template_list:

        template_string = dataset_templates[template].jinja
        template_anser_choices = dataset_templates[template].answer_choices

        entities = {}
        offset = 0

        regex = r"(?:{|{{).*?(?:}}|})"
        matches = re.finditer(regex, template_string, re.DOTALL)
        for idx, m in enumerate(matches, start=1):

            entity_identifier = "{{e{}}}".format(idx)

            start_idx = m.start()
            end_idx = m.end()
            original_entity_len = end_idx - start_idx + 1

            start_idx -= offset
            end_idx -= offset
            offset += (original_entity_len - len(entity_identifier) - 1)

            start_string = template_string[:start_idx]
            end_string = template_string[end_idx:]

            entities[entity_identifier] = {
                "replaces"  : template_string[start_idx:end_idx],
                "start" : start_idx,
                "end"   : start_idx+len(entity_identifier),
            }

            template_string = start_string + entity_identifier + end_string

        template_entities[template] = entities

        template_name = "{} - {} - {}".format(dataset_name, subset_name, template)

        template_string_list.append(template_string)
        template_name_list.append(template_name)

# f = open("entities.json","w")
# f.write(entities)
# f.close()

template_df = pd.DataFrame(
    data={
        "template_name"     : template_name_list,
        "template_string"   : template_string_list,
        },
    )
