import os

import pandas as pd

from src import ROOT_DIR

all_ranks_revised_case = os.path.join(ROOT_DIR, 'test/sandbox_aprior_code_suggestion/all_ranks_without_codes.csv')
all_ranks_revised_case = pd.read_csv(all_ranks_revised_case)

rank_less10 = 10  # less than 10 codes need to review
rank_top3 = 4  # less than 4 code need to review

revised_case_all_info = os.path.join(ROOT_DIR, 'resources/db_data/revised_case_all_info.csv')
revised_case_all_info_df = pd.read_csv(revised_case_all_info)
cw_all_before_revision = revised_case_all_info_df['drg_cost_weight_original'].sum()
cw_all_after_revision = revised_case_all_info_df['drg_cost_weight_revised'].sum()

methods = list()
cw_before_revision = list()
cw_after_revision = list()
cw_before_revision_less10 = list()
cw_after_revision_less10 = list()
cw_before_revision_top3 = list()
cw_after_revision_top3 = list()

num_cases_need_review = list()
num_cases_less10 = list()
num_cases_top3 = list()

methods.append('d2d')
cw_before_revision.append(cw_all_before_revision)
cw_after_revision.append(cw_all_after_revision)
num_cases_need_review.append(revised_case_all_info_df.shape[0])
num_cases_less10.append(revised_case_all_info_df.shape[0])
num_cases_top3.append(revised_case_all_info_df.shape[0])
cw_before_revision_less10.append(0)
cw_after_revision_less10.append(0)

cw_before_revision_top3.append(0)
cw_after_revision_top3.append(0)

rank_cols = [col for col in all_ranks_revised_case.columns if 'rank_' in col]
for rank_col in rank_cols:
    method_name = rank_col.split('_')[3]
    rank_df = all_ranks_revised_case[['CaseId', 'added_ICD'] + [rank_col]]
    rank_df.dropna(subset=rank_col, inplace=True)
    rank_df[rank_col] = rank_df[rank_col].astype(int)

    # calculate all cw can be obtained for each method
    case_ids_all = rank_df["CaseId"].tolist()
    num_suggests = len(set(case_ids_all))
    cases_with_suggests = revised_case_all_info_df[revised_case_all_info_df['case_id'].isin(case_ids_all)]
    cw_before_all = cases_with_suggests['drg_cost_weight_original'].sum()
    cw_after_all = cases_with_suggests['drg_cost_weight_revised'].sum()

    # calculate cs when review less than 10 codes for each method
    rank_less_10_df = rank_df[rank_df[rank_col] < rank_less10]
    case_ids_less10 = rank_less_10_df['CaseId'].tolist()
    num_suggest_less10 = len(set(case_ids_less10))
    # calculate cs when review less than 4 codes for each method
    rank_top3_df = rank_df[rank_df[rank_col] < rank_top3]
    case_ids_top3 = rank_top3_df['CaseId'].tolist()
    num_suggest_top3 = len(set(case_ids_top3))

    # get the case from db and calculate cw (less 10)
    cases_with_suggests_less10 = revised_case_all_info_df[revised_case_all_info_df['case_id'].isin(case_ids_less10)]
    cw_before_less10 = cases_with_suggests_less10['drg_cost_weight_original'].sum()
    cw_after_less10 = cases_with_suggests_less10['drg_cost_weight_revised'].sum()
    # get the case from db and calculate cw (top 4)
    cases_with_suggests_top3 = revised_case_all_info_df[revised_case_all_info_df['case_id'].isin(case_ids_top3)]
    cw_before_top3 = cases_with_suggests_top3['drg_cost_weight_original'].sum()
    cw_after_top3 = cases_with_suggests_top3['drg_cost_weight_revised'].sum()

    methods.append(method_name)
    # cost weight if review all the suggested case with all code
    cw_before_revision.append(cw_before_all)
    cw_after_revision.append(cw_after_all)
    # cost weight if review cases with less than 10 codes suggested
    cw_before_revision_less10.append(cw_before_less10)
    cw_after_revision_less10.append(cw_after_less10)
    # cost weight if review cases with top 3 (less than 4) codes suggested
    cw_before_revision_top3.append(cw_before_top3)
    cw_after_revision_top3.append(cw_after_top3)
    # total number of cases for each type
    num_cases_need_review.append(num_suggests)
    num_cases_less10.append(num_suggest_less10)
    num_cases_top3.append(num_suggest_top3)

cw_methods = pd.DataFrame(list(
    zip(methods, num_cases_need_review, num_cases_less10, num_cases_top3,  cw_before_revision, cw_after_revision,
        cw_before_revision_less10, cw_after_revision_less10, cw_before_revision_top3, cw_after_revision_top3)))

cw_methods.columns = ['method', 'num_cases_need_review', 'num_cases_less10', 'num_cases_top3','cw_before_revision', 'cw_after_revision',
                      'cw_before_revision_less10', 'cw_after_revision_less10',  'cw_before_revision_top3', 'cw_after_revision_top3']
cw_methods['delta_cw'] = cw_methods['cw_after_revision'] - cw_methods['cw_before_revision']
cw_methods['delta_cw_less10'] = cw_methods['cw_after_revision_less10'] - cw_methods['cw_before_revision_less10']
cw_methods['delta_cw_top3'] = cw_methods['cw_after_revision_top3'] - cw_methods['cw_before_revision_top3']

output_result = os.path.join(ROOT_DIR, 'results', 'delta_cost_weight_analysis')
if not os.path.exists(output_result):
    os.makedirs(output_result)
cw_methods.to_csv(os.path.join(output_result, 'delta_cost_weight_analysis_mindbend_apriori.csv'))

