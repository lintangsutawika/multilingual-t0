import re
import json

import numpy as np
import pandas as pd

from promptsource.templates import DatasetTemplates, Template

def add_translated_prompt_templates():

    with open('csv_files/entities.json') as f:
        template_entities_dict = json.load(f)

    translated_prompts_df = pd.read_csv('csv_files/template_translated.csv')
    language_code_list = list(
        set(translated_prompts_df.columns) - {'template_name', 'template_string'}
        )

    dataset_name_dict = {}
    print("####")
    print("Processing translated prompts")
    print("####")
    for _, row in translated_prompts_df.iterrows():
        template_code_name = row['template_name']
        print("Processing {}".format(template_code_name))
        dataset_name, subset_name, *template_name = template_code_name.split(' - ')
        template_name = ' - '.join(template_name)

        dict_key = "{}-{}".format(dataset_name, subset_name)
        if dict_key not in dataset_name_dict:
            dataset_name_dict[dict_key] = 0

        if subset_name == "None":
            subset_name = None

        dataset_templates = DatasetTemplates(
            dataset_name=dataset_name,
            subset_name=subset_name
            )

        try:
            entity_list = template_entities_dict[template_code_name]
        except:
            template_code_name += ' '
            template_name += ' '
            entity_list = template_entities_dict[template_code_name]

        if 'answer_choices' in entity_list:
            answer_choices = entity_list.pop('answer_choices')

        # idx = 0
        for language_code in language_code_list:
            translated_template_string = row[language_code]

            translated_template_string = re.sub(r'(?:\|).*(?:\|)', '|||', translated_template_string)

            regex = r"(?:{).*?(?:})"
            for m in re.finditer(regex, translated_template_string, re.DOTALL):

                _string = m.group()
                _string_fixed = _string.lower()

                if bool(re.search("{e.*?[0-9]}",_string_fixed))==True:
                    _string_fixed = _string_fixed.replace(" ", "")

                translated_template_string = re.sub(_string, _string_fixed, translated_template_string)

            for entity, original_entity in entity_list.items():
                translated_template_string = re.sub(entity, original_entity, translated_template_string)

            prefix = "Answser in English, \n"
            translated_template_string = prefix + translated_template_string
            idx = dataset_name_dict[dict_key]

            new_template_name = "{}-translate-{}".format(template_name, language_code)
            if new_template_name not in dataset_templates.all_template_names:
                print("Adding {} for {}".format(new_template_name, template_code_name))
                new_template = Template(
                    name=new_template_name,
                    jinja=translated_template_string,
                    answer_choices=answer_choices,
                    reference="Translated version of {} in {}".format(template_name, language_code)
                    )
        
                new_template.id = '00000000-0000-0000-0000-'+str(idx).zfill(12)

                dataset_templates.add_template(new_template)
                dataset_name_dict[dict_key] += 1
            else:
                print("Already added {} for {}".format(new_template_name, template_code_name))


if __name__ == '__main__':

    print('Prepare file of encoded prompts for translation')
    dataset_lists_df = pd.read_csv('csv_files/datasets.csv')
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
            answer_choices = dataset_templates[template].answer_choices

            template_name = "{} - {} - {}".format(dataset_name, subset_name, template)

            entities = {'answer_choices': answer_choices}
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

                entities[entity_identifier] = template_string[start_idx:end_idx]
                #     "start" : start_idx,
                #     "end"   : start_idx+len(entity_identifier),
                # }

                template_string = start_string + entity_identifier + end_string

            template_entities[template_name] = entities
            template_string_list.append(template_string)
            template_name_list.append(template_name)

            # template_answer_choices = dataset_templates[template].answer_choices
            # if template_answer_choices is not None:
            #     template_answer_choices = template_answer_choices.split('|||')
            #     template_answer_choices = [re.sub(' ', '', t) for t in template_answer_choices]

            #     for idx, answer_choice in enumerate(template_answer_choices):

            #         template_name = 'Answer Choice - {} - '.format(idx)+template_name

            #         # template_entities[template_name] = answer_choice
            #         template_string_list.append(answer_choice)
            #         template_name_list.append(template_name)

    with open("csv_files/entities.json","w") as f:
        json.dump(template_entities,f)

    template_df = pd.DataFrame(
        data={
            "template_name"     : template_name_list,
            "template_string"   : template_string_list,
            },
        )
    template_df.to_csv('csv_files/template_df.csv', index=False)
