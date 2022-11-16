import os

import awswrangler as wr
import matplotlib
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt

from src.files import load_revised_cases, load_all_rankings
from src.rankings import LABEL_NOT_SUGGESTED, RANKING_LABELS, RANKING_RANGES
from src.schema import case_id_col, suggested_code_rankings_split_col
from src.utils.general_utils import save_figure_to_pdf_on_s3


def calculate_code_ranking_performance(*,
                                       filename_revised_cases: str,
                                       dir_rankings: str,
                                       dir_output: str,
                                       s3_bucket: str = 'code-scout'
                                       ):
    """Calculate the code ranking performance of CodeScout. It uses information from revised cases and the raw output
    of CodeScout. It outputs plots in PDF format on S3.

    @param dir_rankings: This directory contains all ranking results from the recommender systems.
    @param dir_output: Directory to store the results in.
    @param filename_revised_cases: This is the filename to all revised cases we want to compare the rankings to.
    @param s3_bucket: Directory to store the results in.
    """

    # Load the revised cases, which are the ground truth for calculating the performance
    revised_cases = load_revised_cases(filename_revised_cases)
    n_revised_cases = revised_cases.shape[0]

    # Load the rankings from CodeScout
    all_rankings = load_all_rankings(dir_rankings)

    # Collect the code rankings into a list
    code_ranks = list()

    for _, method_name, ranked_cases in all_rankings:
        current_method_code_ranks = list()
        n_revised_cases_without_suggestions = 0

        for _, current_revised_case in revised_cases.iterrows():
            # Get the suggestions for the currently revised case
            case_id = current_revised_case['combined_id']
            ranked_suggestions = ranked_cases[ranked_cases[case_id_col] == case_id][suggested_code_rankings_split_col].values

            # If no suggestions are available, skip this case
            if ranked_suggestions.shape[0] == 0:
                n_revised_cases_without_suggestions += 1
                continue

            # Collect the list of codes which were suggested (and ranked) by CodeScout
            suggested_codes = ranked_suggestions[0]  # There is only one element in the array, which is a list of str

            # Combine all the codes which were added during the revision to upcode the case
            icd_added_list = current_revised_case['ICD_added_split']
            chop_added_list = [x.split(':')[0].replace('.', '') for x in current_revised_case['CHOP_added_split']]
            added_codes = icd_added_list + chop_added_list
            added_codes = list(np.unique(added_codes))

            for code in added_codes:
                if code not in suggested_codes:
                    rank = LABEL_NOT_SUGGESTED
                else:
                    idx = suggested_codes.index(code)
                    rank = idx + 1  # so ranks are 1-based

                current_case_label = rank
                code_rank = [case_id, code, suggested_codes, current_case_label]
                current_method_code_ranks.append(code_rank)

        case_results = pd.DataFrame(np.vstack(current_method_code_ranks), columns=[case_id_col, 'added_ICD', 'ICD_suggested_list', 'rank'])
        case_results.astype({case_id_col: 'string', 'added_ICD': 'string', 'ICD_suggested_list': 'string', 'rank': int})
        code_ranks.append(case_results)

        logger.info(f'{method_name}: {n_revised_cases_without_suggestions}/{n_revised_cases} revised cases had no suggestions')

    # write results to file and get the categorical rankings
    logger.info('Writing single results to s3.')
    code_categorical_ranks = list()
    for i, result in enumerate(code_ranks):
        wr.s3.to_csv(result, os.path.join(dir_output, all_rankings[i][1] + '.csv'), index=False)
        # get categorical rank based on predefined RANKING_RANGES
        in_bins = np.digitize(result['rank'].tolist(), RANKING_RANGES, right=False)
        current_categorical_rank = np.bincount(in_bins)[1:]  # ignore smaller than 1 rankings
        code_categorical_ranks.append(current_categorical_rank)

    # get the index ordering from highest to lowest 1-3 label rank
    logger.info('Plot bar plot and store on s3.')
    ind_best_to_worst_ordering = np.argsort([x[0] for x in code_categorical_ranks])[::-1]
    x = np.arange(len(RANKING_LABELS))
    plt.figure()
    width = 0.9 / len(all_rankings)
    offset_x = np.linspace(0, stop=width*(len(code_ranks)-1), num=len(code_ranks))
    color_map = matplotlib.cm.get_cmap('rainbow')
    color_map_distances = np.linspace(0, 1, len(code_ranks))
    # reverse color to go from green to red
    colors = [color_map(x) for x in color_map_distances]
    for i_loop, i_best_model in enumerate(ind_best_to_worst_ordering):
        bar_heights = code_categorical_ranks[i_best_model]  # replace with np.histogram here
        plt.bar(x + offset_x[i_loop], bar_heights, color=colors[i_loop], width=width, label=all_rankings[i_best_model][1])
    plt.xticks(range(len(RANKING_LABELS)), RANKING_LABELS, rotation=90)
    plt.xlabel('Ranking classes')
    plt.ylabel('Frequency')
    plt.legend(loc='best', fancybox=True, framealpha=0.8, bbox_to_anchor=(1.05, 1.05))
    save_figure_to_pdf_on_s3(plt, s3_bucket, os.path.join(dir_output, 'bar_ranking_classes.pdf'))
    plt.close()

    logger.info('Combining all results to one output file and write to s3.')
    suffix_left = '_'+all_rankings[0][1]
    suffix_right = '_' + all_rankings[1][1]
    combined_ranks = pd.merge(code_ranks[0], code_ranks[1], how='outer', on=['CaseId', 'added_ICD'], suffixes=(suffix_left, suffix_right))
    for i in range(2, len(all_rankings)):
        suffix_right = '_'+all_rankings[i][1]
        combined_ranks = pd.merge(combined_ranks, code_ranks[i], how='outer', on=['CaseId', 'added_ICD'], suffixes=(None, suffix_right))
        combined_ranks.rename(columns={'ICD_suggested_list' : 'ICD_suggested_list_' + all_rankings[i][1], 'rank': 'rank_'+all_rankings[i][1]}, inplace=True)
    combined_ranks = combined_ranks.reindex(['CaseId', 'added_ICD'] + ['rank_' + x[1] for x in all_rankings] + ['ICD_suggested_list_' + x[1] for x in all_rankings], axis=1)
    wr.s3.to_csv(combined_ranks, os.path.join(dir_output, 'all_ranks.csv'), index=False)


if __name__ == '__main__':
    calculate_code_ranking_performance(
        dir_rankings='s3://code-scout/performance-measuring/code_rankings/2022-10-20_filter_comparison_without_targetDRG/',
        dir_output='s3://code-scout/performance-measuring/code_rankings/2022-10-20_filter_comparison_without_targetDRG_results_4-classes/',
        filename_revised_cases='s3://code-scout/performance-measuring/CodeScout_GroundTruthforPerformanceMeasuring.csv',
        s3_bucket='code-scout'
    )
