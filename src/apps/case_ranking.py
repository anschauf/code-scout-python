import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src import venn
from src.files import load_revised_cases, load_all_rankings
from src.schema import case_id_col, prob_most_likely_code_col
from src.utils import save_figure_to_pdf_on_s3


def create_rankings_of_revised_cases(*,
                                     filename_revised_cases: str,
                                     dir_rankings: str,
                                     dir_output: str,
                                     s3_bucket: str = 'code-scout'
                                     ):
    """Calculate the case ranking performance of CodeScout. It uses information from revised cases and the raw output
    of CodeScout. It outputs plots in PDF format on S3.

    @param filename_revised_cases: This is the filename to all revised cases we want to compare the rankings to.
    @param dir_rankings: This directory contains all ranking results from the recommender systems.
    @param dir_output: Directory to store the results in.
    @param s3_bucket: Directory to store the results in.
    """
    rank_col = 'prob_rank'

    top100 = set(np.arange(0, 100))
    top1000 = set(np.arange(0, 1000))

    # Load the revised cases, which are the ground truth for calculating the performance
    revised_cases = load_revised_cases(filename_revised_cases)

    # Load the rankings from CodeScout
    all_rankings = load_all_rankings(dir_rankings)

    cdf_delta_cw = dict()
    num_cases = dict()
    for hospital_year, method_name, rankings in all_rankings:
        # Sort the cases based on probabilities, and add a column indicating the rank
        rankings = rankings.sort_values(by=prob_most_likely_code_col, ascending=False).reset_index(drop=True)
        rankings[rank_col] = rankings[prob_most_likely_code_col].rank(method='min', ascending=False)

        # Perform an inner join between revised cases and ranked results from CodeScout
        revised_cases[case_id_col] = revised_cases['combined_id']
        overlap = pd.merge(revised_cases, rankings, on=case_id_col, how='inner')
        revised_codescout = overlap[[case_id_col, 'CW_old', 'CW_new', rank_col]].sort_values(by=rank_col)

        # Calculate the delta cost-weight
        revised_codescout['delta_CW'] = revised_codescout['CW_new'].astype(float) - revised_codescout['CW_old'].astype(float)

        # The cumsum is the empirical cumulative distribution function (ECDF)
        revised_codescout['cdf'] = revised_codescout['delta_CW'].cumsum()

        x = revised_codescout[rank_col].values
        y = revised_codescout['cdf'].values

        num_cases[method_name] = rankings.shape[0]
        cdf_delta_cw[method_name] = (x, y)

        # Computation of the Venn diagram
        all_cases = set(rankings[rank_col].tolist())
        revised_codescout_overlap = set(revised_codescout[rank_col].tolist())

        labels = venn.get_labels([top100, top1000, all_cases, revised_codescout_overlap],
                                 fill=['number', 'logic'])
        fig, ax = venn.venn4(labels, names=['Top 100', 'Top 1000', 'All cases', 'Revised cases'])
        fig.suptitle(f'Case Ranking Tier ({hospital_year})', fontsize=40)
        save_figure_to_pdf_on_s3(fig, s3_bucket, os.path.join(dir_output, 'case_ranking_plot_venn.pdf'))
        fig.show()

    # Cumulative plot for each method from cdf_delta_cw
    cdf_list = list()
    for method_name, data in cdf_delta_cw.items():
        df = pd.DataFrame(data).transpose()
        df.columns = [rank_col, 'cdf_delta_cw']
        cdf_list.append(df)

    plt.figure()
    for method_name, data in cdf_delta_cw.items():
        ranks, cdf = cdf_delta_cw[method_name]
        n_cases = num_cases[method_name]
        x = [0] + ranks + [n_cases]
        y = [0] + cdf + [cdf[-1]]
        plt.step(x, y, where='post', label=method_name)
    plt.xlabel("# cases")
    plt.ylabel("delta CW")
    plt.title("Cumulative distribution of delta cost weight (CW_delta)")
    plt.legend()
    save_figure_to_pdf_on_s3(plt, s3_bucket, os.path.join(dir_output, 'case_ranking_plot_cdf.pdf'))


if __name__ == '__main__':
    create_rankings_of_revised_cases(
        filename_revised_cases="s3://code-scout/performance-measuring/CodeScout_GroundTruthforPerformanceMeasuring.csv",
        dir_rankings='s3://code-scout/performance-measuring/code_rankings/2022-09-14_first-filter-comparison_ksw/',
        dir_output="s3://code-scout/performance-measuring/case_rankings/DRG_tree/revisions/ksw_2019_2_case_ranking_tier_plots/",
        s3_bucket='code-scout'
    )
