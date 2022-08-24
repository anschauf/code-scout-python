import os
from os.path import basename, splitext

import awswrangler as wr
import matplotlib
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt

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
    logger.info(f'Reading revised cases from {filename_revised_cases} ...')
    revised_cases = wr.s3.read_csv(filename_revised_cases)
    logger.info(f'Read {revised_cases.shape[0]} rows')

    revised_cases['ICD_added_split'] = revised_cases['ICD_added'].apply(split_codes)
    revised_cases['CHOP_added_split'] = revised_cases['CHOP_added'].apply(split_codes)
    revised_cases['CHOP_dropped_split'] = revised_cases['CHOP_dropped'].apply(split_codes)

    # load rankings and store them in a tuple
    logger.info(f'Listing files in {dir_rankings} ...')
    all_ranking_filenames = wr.s3.list_objects(dir_rankings)

    all_rankings = list()
    for filename in all_ranking_filenames:
        logger.info(f'Reading {filename} ...')
        temp_data = wr.s3.read_csv(filename)
        temp_method_name = splitext(basename(filename))[0]
        all_rankings.append((temp_method_name, temp_data))

    code_ranks = []
    label_not_suggested = 'not suggest'
    ranking_labels = ['rank_1_3', 'rank_4_6', 'rank_7_9', 'rank_10', 'rank_not_suggest']
    for method_name, rankings in all_rankings:
        rankings['suggested_codes_pdx_split'] = rankings['suggested_codes_pdx'].apply(split_codes)
        revised_cases['ICD_added_split'] = revised_cases['ICD_added'].apply(split_codes)

        current_method_code_ranks = list()
        for case_index in range(revised_cases.shape[0]):
            current_case = revised_cases.iloc[case_index]
            case_id = current_case['CaseId']
            icd_added_list = current_case['ICD_added_split']
            ranking_case_id = rankings['case_id'].tolist()

            # find matching case id in rankings if present
            # if not present, skip
            if case_id in ranking_case_id:
                # TODO check whether this case id is unique, if not discard case because we can not indentify it uniquely, because now we just take the first one, even if we have more
                icd_suggested_list = rankings[rankings['case_id'] == case_id]['suggested_codes_pdx_split'].values[0]
            else:
                continue

            # if present split icd_added field in current_case object into single diagnoses, e.g. K5722|E870 => ['K5722',  'E870']
            # use .split('|')

            # find revised diagnoses in current ranking after also here splitting the diagnoses like before with the revised case
            # if diagnosis is present, find index where its ranked and classify it in one of the ranking labels
            # 1-3, 4-6, 7-9, 10+
            # if not present add to not suggested label
            for icd in icd_added_list:
                if icd not in icd_suggested_list:
                    rank = label_not_suggested
                else:
                    idx = icd_suggested_list.index(icd)  # error was here
                    rank = idx + 1

            current_case_label = get_categorical_ranks(rank, label_not_suggested)
            code_rank = [case_id, icd, icd_suggested_list] + list(current_case_label)
            current_method_code_ranks.append(code_rank)

        code_ranks.append(pd.DataFrame(np.vstack(current_method_code_ranks), columns=['case_id', 'added_ICD', 'ICD_suggested_list'] + ranking_labels))

    # write results to file
    for i, result in enumerate(code_ranks):
        wr.s3.to_csv(result, os.path.join(dir_output, all_rankings[i][0] + '.csv'), index=False)

    # get the index ordering from highest to lowest 1-3 label rank
    ind_best_to_worst_ordering = np.argsort([x[ranking_labels[0]].sum() for x in code_ranks])[::-1]

    X = np.arange(len(ranking_labels))
    plt.figure()
    width = 0.25
    offset_x = np.linspace(0, stop=width*(len(code_ranks)-1), num=len(code_ranks))
    color_map = matplotlib.cm.get_cmap('rainbow')
    color_map_distances = np.linspace(0, 1, len(code_ranks))
    # reverse color to go from green to red
    colors = [color_map(x) for x in color_map_distances]
    for i_loop, i_best_model in enumerate(ind_best_to_worst_ordering):
        bar_heights = code_ranks[i_best_model][ranking_labels].sum(axis=0)
        plt.bar(X + offset_x[i_loop], bar_heights, color=colors[i_loop], width=width, label=all_rankings[i_best_model][0])
    plt.xticks(range(5), ranking_labels, rotation=90)
    plt.xlabel('Ranking classes')
    plt.ylabel('Frequency')
    plt.legend(loc='best', fancybox=True, framealpha=0.8)
    save_figure_to_pdf_on_s3(plt, s3_bucket, os.path.join(dir_output, 'bar_ranking_classes.pdf'))
    plt.close()


if __name__ == '__main__':
    calculate_performance(
        dir_rankings='s3://code-scout/performance-measuring/mock_rankings/',
        dir_output='s3://code-scout/performance-measuring/mock_rankings_results/',
        filename_revised_cases='s3://code-scout/performance-measuring/revised_evaluation_cases.csv',
        s3_bucket='code-scout'
    )
