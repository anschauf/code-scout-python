import os.path

import pandas as pd
from loguru import logger

from src import ROOT_DIR
from test.sandbox_aprior_code_suggestion.utils import clean_alt_list


num_condition_code_apriori = 'condition_1_drg'
folder_name = f'revised_cases_ksw2020_combined_suggestions_apriori_mindbend_{num_condition_code_apriori}'
output_dir = os.path.join(ROOT_DIR, 'test/sandbox_aprior_code_suggestion', folder_name)

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

mindbend_suggestions = pd.read_csv(os.path.join(ROOT_DIR, 'test/sandbox_aprior_code_suggestion/mind_bend_codes_list_ksw_2020.csv'), dtype='string[pyarrow]')
dir_apriori_suggestions = os.path.join(ROOT_DIR, 'test/sandbox_aprior_code_suggestion/revised_cases_ksw2020_pd_chops_confidence_condition_three')
revised_case_ksw2020_all_info = pd.read_csv(os.path.join(ROOT_DIR, 'test/sandbox_aprior_code_suggestion/revised_case_ksw_2020_all_info.csv'), dtype='string[pyarrow]')
revised_case_ksw2020case_id = revised_case_ksw2020_all_info['case_id'].tolist()
mindbend_suggestions_revised_cases = mindbend_suggestions[mindbend_suggestions['case_id'].isin(revised_case_ksw2020case_id)]

# mindbend_suggestions.columns: 'case_id', 'mind_bend_suggested_icds', 'mind_bend_suggested_chops'
mindbend_suggestions_revised_cases['mind_bend_suggested_icds'] = mindbend_suggestions_revised_cases['mind_bend_suggested_icds'].apply(clean_alt_list)
mindbend_suggestions_revised_cases['mind_bend_suggested_chops'] = mindbend_suggestions_revised_cases['mind_bend_suggested_chops'].apply(clean_alt_list)
for file_name in os.listdir(dir_apriori_suggestions):

    if file_name.endswith(f'{num_condition_code_apriori}.csv'): # get the suggestions based on three code
        file_name_ls = file_name.replace("'", '').split('=')
        case_id = file_name_ls[1].split('_')[0]
        hospital_name = file_name_ls[2].split('_')[0]
        year = int(file_name_ls[3].split('_')[0])

        apriori_suggestions_case = pd.read_csv(os.path.join(dir_apriori_suggestions, file_name))
        mind_bend_suggestions_icds = mindbend_suggestions_revised_cases[mindbend_suggestions_revised_cases['case_id'] == case_id]['mind_bend_suggested_icds'].tolist()[0]
        mind_bend_suggestions_chops = mindbend_suggestions_revised_cases[mindbend_suggestions_revised_cases['case_id'] == case_id]['mind_bend_suggested_chops'].tolist()[0]

        apriori_suggestions_case['consequents_ls'] = apriori_suggestions_case['consequents_ls'].apply(clean_alt_list)
        apriori_suggestions_case['contains_icds_mindbend'] = apriori_suggestions_case['consequents_ls'].apply(lambda x: len(set(x).intersection(set(mind_bend_suggestions_icds))) >= 1)
        apriori_suggestions_case['contains_chops_mindbend'] = apriori_suggestions_case['consequents_ls'].apply(lambda x: len(set(x).intersection(set(mind_bend_suggestions_chops))) >= 1)

        #  apriori auggestions contain either icds or chops from mindbend
        combined_suggestions_apriori_mindbend = apriori_suggestions_case[(apriori_suggestions_case['contains_icds_mindbend']==True) | (apriori_suggestions_case['contains_chops_mindbend']==True)]
        num_suggestions_icds = combined_suggestions_apriori_mindbend.shape[0]
        # save all suggestions both from apriori and mindbend
        if num_suggestions_icds >=1:
            file_path = os.path.join(output_dir, f'{case_id}_combined_suggestions_apriori_mindbend_{num_condition_code_apriori}.csv')
            combined_suggestions_apriori_mindbend.to_csv(file_path)
            logger.info(f'Successfully saved {num_suggestions_icds} combined suggestions from mindbend and apriori for {case_id=}')

        else:
            logger.info(
                f'No combined suggestions from mindbend and apriori for {case_id=}')
