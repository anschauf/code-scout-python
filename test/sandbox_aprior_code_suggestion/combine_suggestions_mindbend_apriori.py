import os.path
from collections import Counter
from itertools import chain

import pandas as pd
from loguru import logger

from src import ROOT_DIR
from test.sandbox_aprior_code_suggestion.utils import clean_alt_list
from test.sandbox_aprior_code_suggestion.apriori_related_functions import combine_apriori_mindbend

from test.sandbox_model_case_predictions.data_handler import load_data_single_file
from test.sandbox_model_case_predictions.utils import get_revised_case_ids


num_condition_code_apriori = 'condition_all_drg'
# min_confidence = 0.25

# hospital = "Kantonsspital Winterthur"
file_name = 'KSW_2020.json'
# hospital_year = "KSW_2020"
year = 2020

revised_case_ksw_2020 = False
revised_case_all = True
min_confidence = 0.20
# get the hospital and year
dir_data = os.path.join(ROOT_DIR, 'resources', 'data')
REVISED_CASE_IDS_FILENAME = os.path.join(ROOT_DIR, 'resources', 'data', 'revised_case_ids.csv')

data_single_hospital = load_data_single_file(dir_data=dir_data, file_name=file_name)
revised_cases_in_data = get_revised_case_ids(data_single_hospital, REVISED_CASE_IDS_FILENAME, overwrite=False)



if revised_case_ksw_2020:
    dir_apriori_suggestions = os.path.join(ROOT_DIR,
                                           f'results/sandbox_aprior_code_suggestion/revised_case_ksw2020_apriori_confidence_0.25')

    folder_name = f'revised_case_ksw2020_apriori_confidence_0.25_combined_suggestions_apriori_mindbend_{num_condition_code_apriori}'
    output_dir = os.path.join(ROOT_DIR, 'results/aprior_code_suggestions', folder_name)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    revised_cases_ksw2020 = revised_cases_in_data[(revised_cases_in_data["is_revised"] == 1) and (revised_cases_in_data["hospital"] == 'KSW') and (revised_cases_in_data["dischargeYear"] == 2020)]
    case_ids_ksw2020 = revised_cases_ksw2020['id'].tolist()

    mindbend_suggestions = pd.read_csv(
        os.path.join(ROOT_DIR, 'test/sandbox_aprior_code_suggestion/mind_bend_codes_list_reviewed_cases.csv'), sep=',')
    mindbend_suggestions = mindbend_suggestions[
        mindbend_suggestions['case_id'].isin(case_ids_ksw2020)]

    mindbend_suggestions.dropna(axis=1, inplace=True)
    mindbend_suggestions.drop_duplicates(subset='case_id', inplace=True)
    # mindbend_suggestions.columns: 'case_id', 'mind_bend_suggested_icds', 'mind_bend_suggested_chops'
    mindbend_suggestions['mind_bend_suggested_icds'] = mindbend_suggestions[
        'mind_bend_suggested_icds'].apply(clean_alt_list)
    mindbend_suggestions['mind_bend_suggested_chops'] = mindbend_suggestions[
        'mind_bend_suggested_chops'].apply(clean_alt_list)

    summary_suggestions_all = list()
    for apriori_hospital_year in os.listdir(dir_apriori_suggestions):
        if apriori_hospital_year.startswith('revised_case_'):
            hospital = apriori_hospital_year.split('_')[2]
            year = apriori_hospital_year.split('_')[3]

            apriori_hospital_year_path = os.path.join(dir_apriori_suggestions, apriori_hospital_year)
            summary_suggestions = combine_apriori_mindbend(apriori_hospital_year_path, mindbend_suggestions,
                                                           num_condition_code_apriori, output_dir)
            summary_suggestions_all.append(summary_suggestions)
        summary_suggestions_all_df = pd.concat(summary_suggestions_all)
        summary_suggestions_all_df.to_csv(os.path.join(output_dir, 'summary_sugggestions_revised_case_df.csv'))


if revised_case_all:
    dir_apriori_suggestions = os.path.join(ROOT_DIR,
                                           'results/aprior_code_suggestions')

    folder_name = f'revised_case_apriori_confidence_0.25_combined_suggestions_apriori_mindbend_{num_condition_code_apriori}'
    output_dir = os.path.join(ROOT_DIR, 'results/aprior_code_suggestions', folder_name)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    revised_cases_only = revised_cases_in_data[revised_cases_in_data["is_revised"] == 1]
    revised_cases_only_case_id = revised_cases_only['id'].tolist()
    # mindbend suggestions
    mindbend_suggestions = pd.read_csv(
        os.path.join(ROOT_DIR, 'test/sandbox_aprior_code_suggestion/mind_bend_codes_list_reviewed_cases.csv'), sep=',')
    mindbend_suggestions = mindbend_suggestions[
        mindbend_suggestions['case_id'].isin(revised_cases_only_case_id)]
    mindbend_suggestions.dropna(axis=1, inplace=True)
    mindbend_suggestions.drop_duplicates(subset='case_id', inplace=True)
    # mindbend_suggestions.columns: 'case_id', 'mind_bend_suggested_icds', 'mind_bend_suggested_chops'
    mindbend_suggestions['mind_bend_suggested_icds'] = mindbend_suggestions[
        'mind_bend_suggested_icds'].apply(clean_alt_list)
    mindbend_suggestions['mind_bend_suggested_chops'] = mindbend_suggestions[
        'mind_bend_suggested_chops'].apply(clean_alt_list)

    summary_suggestions_all = list()
    summary_suggestions_apriori_all = list()
    for apriori_hospital_year in os.listdir(dir_apriori_suggestions):
        if apriori_hospital_year.startswith(
                'revised_case_') and apriori_hospital_year.find('_combined_suggestions_apriori_mindbend') == -1:
            hospital = apriori_hospital_year.split('_')[2]
            year = int(apriori_hospital_year.split('_')[3])

            apriori_hospital_year_path = os.path.join(dir_apriori_suggestions, apriori_hospital_year)
            summary_suggestions, summary_suggestions_apriori = combine_apriori_mindbend(apriori_hospital_year_path,
                                                                                        mindbend_suggestions,
                                                                                        num_condition_code_apriori,
                                                                                        output_dir)
            summary_suggestions_all.append(summary_suggestions)
            summary_suggestions_apriori_all.append(summary_suggestions_apriori)

    summary_suggestions_all_df = pd.concat(summary_suggestions_all)
    summary_suggestions_apriori_all_df = pd.concat(summary_suggestions_apriori_all)

    # delete cases has not suggestions
    summary_suggestions_all_df = summary_suggestions_all_df[~(summary_suggestions_all_df['suggested_codes_pdx'] == '')]
    summary_suggestions_apriori_all_df = summary_suggestions_apriori_all_df[~(summary_suggestions_apriori_all_df['suggested_codes_pdx'] == '')]

    summary_suggestions_all_df.to_csv(os.path.join(output_dir, 'summary_sugggestions_combined_revised_case_df.csv'))
    summary_suggestions_apriori_all_df.to_csv(os.path.join(output_dir, 'summary_sugggestions_apriori_revised_case_df.csv'))


else:
    topn = 100
    dir_apriori_suggestions = os.path.join(ROOT_DIR,
                                           f'results/aprior_code_suggestions')
    folder_name = f'top100_KSW_2020_apriori_min_confidence=0.25_combined_suggestions_apriori_mindbend_{num_condition_code_apriori}'
    output_dir = os.path.join(ROOT_DIR, 'results/aprior_code_suggestions', folder_name)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    # case_id for top 500 cases from random forest
    case_rf = pd.read_csv(os.path.join(ROOT_DIR,
                                       'test/sandbox_aprior_code_suggestion/n-estimator-1000_max-depth-9_min-sample-leaf-400_min-samples-split-1_KSW-2020.csv'),
                          sep=";")
    case_rf.sort_values(by='UpcodingConfidenceScore', inplace=True, ascending=False)
    # exclude get the revised and reviewed cases
    cases_in_data_hospital = revised_cases_in_data[
        (revised_cases_in_data['hospital'] == 'KSW') & (revised_cases_in_data['dischargeYear'] == 2020)]
    revised_reviewed_cases_hospital = cases_in_data_hospital[
        (cases_in_data_hospital['is_reviewed'] == 1) | (cases_in_data_hospital['is_revised'] == 1)]

    # only pick M100 and M200
    # cases_in_data_hospital_m100_m200 = data_single_hospital[
    #     (data_single_hospital['Hauptkostenstelle'] == 'M100') | (data_single_hospital['Hauptkostenstelle'] == 'M200')]
    # cases_in_data_hospital_m100_m200_case_ids = cases_in_data_hospital_m100_m200['id'].tolist()

    revised_reviewed_cases_hospital_case_ids = revised_reviewed_cases_hospital['id'].tolist()
    case_ids_rf = case_rf['CaseId'].astype(str).tolist()

    # get case_ids (not reviewed and not reivsed)
    case_ids_not_reviewed_revised = [case_id for case_id in case_ids_rf if
                                     (case_id not in revised_reviewed_cases_hospital_case_ids)]
    # get case_ids from m100 and m200
    # case_ids_m100_m200 = [case_id for case_id in case_ids_not_reviewed_revised if (case_id in cases_in_data_hospital_m100_m200_case_ids)]

    # pick up top n for suggestions
    # case_ids = case_ids_m100_m200[:topn]
    case_ids = case_ids_not_reviewed_revised[:topn]

    mindbend_suggestions = mindbend_suggestions[
        mindbend_suggestions['case_id'].isin(case_ids)]

    summary_suggestions_all = list()
    summary_suggestions_apriori_all = list()
    for apriori_hospital_year in os.listdir(dir_apriori_suggestions):
        if apriori_hospital_year.startswith(
                'top100_KSW_2020_') and apriori_hospital_year.find('_combined_suggestions_apriori_mindbend')==-1:
            hospital = apriori_hospital_year.split('_')[2]
            year = int(apriori_hospital_year.split('_')[3])

            apriori_hospital_year_path = os.path.join(dir_apriori_suggestions, apriori_hospital_year)
            summary_suggestions, summary_suggestions_apriori = combine_apriori_mindbend(apriori_hospital_year_path, mindbend_suggestions,
                                                           num_condition_code_apriori, output_dir)
            summary_suggestions_all.append(summary_suggestions)
            summary_suggestions_apriori_all.append(summary_suggestions_apriori)

    summary_suggestions_all_df = pd.concat(summary_suggestions_all)
    summary_suggestions_apriori_all_df = pd.concat(summary_suggestions_apriori_all)

    # delete cases has not suggestions
    summary_suggestions_all_df = summary_suggestions_all_df[~(summary_suggestions_all_df['suggested_codes_pdx'] == '')]
    summary_suggestions_all_df.to_csv(os.path.join(output_dir, 'top100_KSW_2020_suggestion_summary.csv'))

