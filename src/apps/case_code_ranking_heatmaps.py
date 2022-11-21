import os.path

import awswrangler as wr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.general_utils import save_figure_to_pdf_on_s3


def plot_heatmap_comparing_all_case_code_rankings(*,
                                                  dir_results: str,
                                                  s3_bucket: str = 'code-scout'
                                                  ) -> object:
    """ This function generates clustermaps of all case code rankings. The output contains the four following clustermaps:
            - clustermap on non-standardised rankings on all cases
            - clustermap on log-standardised rankings on all cases
            - clustermap on non-standardised rankings with threshold on rank 10
            - clustermap on log-standardised rankings with threshold on rank 10

    @param dir_results: The output directory.
    @param s3_bucket: The S3 bucket.
    """
    all_files = wr.s3.list_objects(dir_results)
    all_files = [x for x in all_files if x.endswith(".csv") and not x.endswith('all_ranks.csv')]

    all_results = list()
    for file in all_files:
        result = wr.s3.read_csv(file, dtype='string')
        result = result.astype({'rank':int})
        result['CaseIdAddedICD'] = result['CaseId'] + '_' + result['added_ICD']
        all_results.append((result, os.path.basename(file)))

    unique_caseId_added_ICD_pairs = np.unique(np.concatenate([x[0]['CaseIdAddedICD'].values for x in all_results]))
    max_rank = np.max(np.concatenate([x[0]['rank'].values for x in all_results]))
    ranking_comparison = np.full((len(unique_caseId_added_ICD_pairs), len(all_results)), max_rank, dtype=int)
    col_name = list()
    for col_index, result in enumerate(all_results):
        result_df = result[0]
        col_name.append(result[1].replace('.csv', ''))

        for row_index, id in enumerate(unique_caseId_added_ICD_pairs):
            if id in result_df['CaseIdAddedICD'].values:
                ind_id = np.where(result_df['CaseIdAddedICD'].values == id)[0][0]
                ranking_comparison[row_index, col_index] = result_df['rank'].values[ind_id]

    plt.figure()
    sns.clustermap(pd.DataFrame(ranking_comparison, columns=col_name, index=unique_caseId_added_ICD_pairs), yticklabels=False, col_cluster=True)
    save_figure_to_pdf_on_s3(plt, s3_bucket, os.path.join(dir_results, 'clustermap_code-ranks.pdf'))
    plt.close()

    plt.figure()
    sns.clustermap(pd.DataFrame(np.log(ranking_comparison), columns=col_name, index=unique_caseId_added_ICD_pairs), yticklabels=False, col_cluster=True)
    save_figure_to_pdf_on_s3(plt, s3_bucket, os.path.join(dir_results, 'clustermap_code-ranks_log.pdf'))
    plt.close()

    ranking_comparison[ranking_comparison>10] = max_rank

    plt.figure()
    sns.clustermap(pd.DataFrame(ranking_comparison, columns=col_name, index=unique_caseId_added_ICD_pairs), yticklabels=False, col_cluster=True)
    save_figure_to_pdf_on_s3(plt, s3_bucket, os.path.join(dir_results, 'clustermap_code-ranks_threshold_rank10.pdf'))
    plt.close()

    plt.figure()
    sns.clustermap(pd.DataFrame(np.log(ranking_comparison), columns=col_name, index=unique_caseId_added_ICD_pairs), yticklabels=False, col_cluster=True)
    save_figure_to_pdf_on_s3(plt, s3_bucket, os.path.join(dir_results, 'clustermap_code-ranks_threshold_rank10_log.pdf'))
    plt.close()


if __name__ == '__main__':
    plot_heatmap_comparing_all_case_code_rankings(dir_results='s3://code-scout/performance-measuring/code_rankings/2022-10-20_filter_comparison_without_targetDRG_results_4-classes/')
