import os

import awswrangler as wr
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.files import load_revised_cases, load_all_rankings
from src.rankings import LABEL_NOT_SUGGESTED, RANKING_LABELS
from src.utils import get_categorical_ranks, save_figure_to_pdf_on_s3, split_codes


def calculate_performance(*,
                          filename_revised_cases: str,
                          dir_rankings: str,
                          dir_output: str,
                          s3_bucket: str = 'code-scout'
                          ):
    """

    @param dir_rankings: This directory contains all ranking results from the recommender systems.
    @param dir_output: Directory to store the results in.
    @param filename_revised_cases: This is the filename to all revised cases we want to compare the rankings to.
    @param s3_bucket: Directory to store the results in.

    @return:
    """
    revised_cases = load_revised_cases(filename_revised_cases)
    all_rankings = load_all_rankings(dir_rankings)

    icd_code_ranks = list()
    chop_code_ranks = list()

    for method_name, rankings in all_rankings:
        ranking_case_id = rankings['case_id'].tolist()

        current_method_code_ranks = list()
        for case_index in range(revised_cases.shape[0]):
            current_case = revised_cases.iloc[case_index]
            case_id = current_case['CaseId']

            icd_added_list = current_case['ICD_added_split']
            chop_added_list = current_case['CHOP_added_split']
            chop_dropped_list = current_case['CHOP_dropped_split']

            # 1. Lookup in the DataFrame, and get the array. Then, if the array is empty, skip this case ID
            # rankings[rankings['case_id'] == case_id]['suggested_codes_pdx_split'].values

            # 2. Lookup in the list of ranked Case IDs with numpy. Then, if the array is empty, skip this case ID
            # indices = np.where(rankings['case_id'] == case_id)[0]
            # if indices.shape[0] == 0:
            #   continue
            # else:
            # rankings.loc[indices[0]]['suggested_codes_pdx_split'].values[0]

            # 3. Get the index of the case id in the list. Add a try / except to catch the ValueError
            # ranking_case_id.index(case_id)

            # find matching case id in rankings if present
            # if not present, skip
            ranked_suggestions = rankings[rankings['case_id'] == case_id]['suggested_codes_pdx_split'].values

            if ranked_suggestions.shape[0] > 0:
                icd_suggested_list = ranked_suggestions[0]  # There is only one element in the array, which is a list of str
            else:
                continue

            # find revised diagnoses in current ranking
            # if diagnosis is present, find index where it's ranked and classify it in one of the ranking labels
            # 1-3, 4-6, 7-9, 10+
            # if not present add to not suggested label

            for icd in icd_added_list:
                if icd not in icd_suggested_list:
                    rank = LABEL_NOT_SUGGESTED
                else:
                    idx = icd_suggested_list.index(icd)
                    rank = idx + 1  # so ranks are 1-based

                current_case_label = get_categorical_ranks(rank)
                code_rank = [case_id, icd, icd_suggested_list] + list(current_case_label)
                current_method_code_ranks.append(code_rank)

        icd_code_ranks.append(pd.DataFrame(np.vstack(current_method_code_ranks), columns=['case_id', 'added_ICD', 'ICD_suggested_list'] + RANKING_LABELS))

    # write results to file
    for i, result in enumerate(icd_code_ranks):
        wr.s3.to_csv(result, os.path.join(dir_output, all_rankings[i][0] + '.csv'), index=False)

    # get the index ordering from highest to lowest 1-3 label rank
    ind_best_to_worst_ordering = np.argsort([x[RANKING_LABELS[0]].sum() for x in icd_code_ranks])[::-1]

    X = np.arange(len(RANKING_LABELS))
    plt.figure()
    width = 0.25
    offset_x = np.linspace(0, stop=width*(len(icd_code_ranks)-1), num=len(icd_code_ranks))
    color_map = matplotlib.cm.get_cmap('rainbow')
    color_map_distances = np.linspace(0, 1, len(icd_code_ranks))
    # reverse color to go from green to red
    colors = [color_map(x) for x in color_map_distances]
    for i_loop, i_best_model in enumerate(ind_best_to_worst_ordering):
        bar_heights = icd_code_ranks[i_best_model][RANKING_LABELS].sum(axis=0)
        plt.bar(X + offset_x[i_loop], bar_heights, color=colors[i_loop], width=width, label=all_rankings[i_best_model][0])
    plt.xticks(range(5), RANKING_LABELS, rotation=90)
    plt.xlabel('Ranking classes')
    plt.ylabel('Frequency')
    plt.legend(loc='best', fancybox=True, framealpha=0.8)
    save_figure_to_pdf_on_s3(plt, s3_bucket, os.path.join(dir_output, 'bar_ranking_classes.pdf'))
    plt.close()


if __name__ == '__main__':
    calculate_performance(
        dir_rankings='s3://code-scout/performance-measuring/rankings/mock_rankings/',
        dir_output='s3://code-scout/performance-measuring/rankings/mock_rankings_results/',
        filename_revised_cases='s3://code-scout/performance-measuring/revised_evaluation_cases.csv',
        s3_bucket='code-scout'
    )
