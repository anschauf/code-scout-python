import os

import awswrangler as wr
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.files import load_revised_cases, load_all_rankings
from src.rankings import LABEL_NOT_SUGGESTED, RANKING_LABELS, RANKING_RANGES
from src.utils import save_figure_to_pdf_on_s3


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
    # chop_code_ranks = list()

    for _, _, rankings in all_rankings:
        # ranking_case_id = rankings['case_id'].tolist()

        current_method_code_ranks = list()
        for case_index in range(revised_cases.shape[0]):
            current_case = revised_cases.iloc[case_index]
            case_id = current_case['combined_id']

            icd_added_list = current_case['ICD_added_split']
            # chop_added_list = current_case['CHOP_added_split']
            # chop_dropped_list = current_case['CHOP_dropped_split']

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
            # skip if case id not present
            ranked_suggestions = rankings[rankings['case_id'] == case_id]['suggested_code_rankings_split'].values

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

                current_case_label = rank
                code_rank = [case_id, icd, icd_suggested_list, current_case_label]
                current_method_code_ranks.append(code_rank)

        icd_code_ranks.append(pd.DataFrame(np.vstack(current_method_code_ranks), columns=['case_id', 'added_ICD', 'ICD_suggested_list', 'rank']))

    # write results to file and get the categorical rankings
    icd_code_categorical_ranks = list()
    for i, result in enumerate(icd_code_ranks):
        wr.s3.to_csv(result, os.path.join(dir_output, all_rankings[i][0] + '.csv'), index=False)
        # get categorical rank based on predefined RANKING_RANGES
        in_bins = np.digitize(result['rank'].tolist(), RANKING_RANGES, right=False)
        current_categorical_rank = np.bincount(in_bins)[1:]  # ignore smaller than 1 rankings
        icd_code_categorical_ranks.append(current_categorical_rank)

    # get the index ordering from highest to lowest 1-3 label rank
    ind_best_to_worst_ordering = np.argsort([x[0] for x in icd_code_categorical_ranks])[::-1]

    X = np.arange(len(RANKING_LABELS))
    plt.figure()
    width = 0.9 / len(all_rankings)
    offset_x = np.linspace(0, stop=width*(len(icd_code_ranks)-1), num=len(icd_code_ranks))
    color_map = matplotlib.cm.get_cmap('rainbow')
    color_map_distances = np.linspace(0, 1, len(icd_code_ranks))
    # reverse color to go from green to red
    colors = [color_map(x) for x in color_map_distances]
    for i_loop, i_best_model in enumerate(ind_best_to_worst_ordering):
        bar_heights = icd_code_categorical_ranks[i_best_model]  # replace with np.histogram here
        plt.bar(X + offset_x[i_loop], bar_heights, color=colors[i_loop], width=width, label=all_rankings[i_best_model][1])
    plt.xticks(range(len(RANKING_LABELS)), RANKING_LABELS, rotation=90)
    plt.xlabel('Ranking classes')
    plt.ylabel('Frequency')
    plt.legend(loc='best', fancybox=True, framealpha=0.8, bbox_to_anchor=(1.05, 0.6))
    save_figure_to_pdf_on_s3(plt, s3_bucket, os.path.join(dir_output, 'bar_ranking_classes.pdf'))
    plt.close()


if __name__ == '__main__':
    calculate_performance(
        dir_rankings='s3://code-scout/performance-measuring/code_rankings/2022-09-07_first-filter-comparison/',
        dir_output='s3://code-scout/performance-measuring/code_rankings/2022-09-07_first-filter-comparison_results_4-classes/',
        filename_revised_cases='s3://code-scout/performance-measuring/CodeScout_GroundTruthforPerformanceMeasuring.csv',
        s3_bucket='code-scout'
    )
