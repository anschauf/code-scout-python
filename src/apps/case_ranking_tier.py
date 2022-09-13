import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from src import venn

from src.files import load_revised_cases, load_code_scout_results
from src.utils import save_figure_to_pdf_on_s3




def create_rankings_of_revised_cases(
        filename_revised_cases: str,
        filename_codescout_results: str,
        dir_output: str,
        s3_bucket: str = 'code-scout'):
    """

    @param dir_rankings: This directory contains all ranking results from the recommender systems.
    @param dir_output: Directory to store the results in.
    @param filename_revised_cases: This is the filename to all revised cases we want to compare the rankings to.
    @param s3_bucket: Directory to store the results in.

    @return:
    """

    # load revision data from DtoD
    revised_cases = load_revised_cases(filename_revised_cases)
    # load the KSW2019 data from Codescout
    codescout_rankings = load_code_scout_results(filename_codescout_results)

    cdf_delt_cw = dict()
    num_cases = dict()
    for hospital_year, method_name, rankings in codescout_rankings:
        # sort the codescout_rankings based on probabilities and get the caseID from codescout_rankings as list
        rankings.sort_values(by='prob_most_likely_code', ascending=False)
        rankings['prob_rank'] = np.arange(1, len(rankings)+1)

        # caseid_codescout = rankings['case_id'].tolist()

        # get the caseid from revised cases as a list
        # caseid_revised = revised_cases['combined_id'].tolist()

        # go to revision cases
        # based on caseID get the index from the Codescout data
        # iterate through revision cases (for loop) doing this:
        #   check if caseID (of DtoD revision data) is present in the Codescout suggestions data
        #   if yes, return the row index (i.e. position or rank bucket)
        # if two cases (two identical CaseID) present in Codescout suggestions -> ignore the case at the moment

        revised_cases['case_id'] = revised_cases['combined_id']

        overlap = pd.merge(revised_cases, rankings, on='case_id', how='inner')

        revised_codescout = overlap[['case_id', 'CW_old', 'CW_new', 'prob_rank']].sort_values(by='prob_rank')

        revised_codescout['delta_CW'] = revised_codescout['CW_new'].astype(float) - revised_codescout['CW_old'].astype(float)

        revised_codescout['cdf'] = revised_codescout['delta_CW'].cumsum()

        x = revised_codescout['prob_rank'].tolist()
        y = revised_codescout['cdf'].tolist()

        num_cases[method_name] = rankings.shape[0]
        cdf_delt_cw[method_name] = [x, y]

        # Computation of the Venn diagram
        top100 = set(np.arange(0, 100))
        top1000 = set(np.arange(0, 1000))
        all_cases = set(rankings['prob_rank'].tolist())
        revised_codescout_overlap = set(revised_codescout['prob_rank'].tolist())

        labels = venn.get_labels([top100, top1000, all_cases, revised_codescout_overlap],
                                 fill=['number', 'logic'])
        fig, ax = venn.venn4(labels, names=['Top 100', 'Top 1000', 'All cases', 'Revised cases'])
        fig.suptitle(f'Case Ranking Tier ({hospital_year})', fontsize=40)
        save_figure_to_pdf_on_s3(fig, s3_bucket, os.path.join(dir_output, 'case_ranking_plot_venn.pdf'))
        # fig.savefig(f'case_ranking_{hospital_year}.png', bbox_inches='tight')
        fig.show()

    # Cumulative plot for each methods from cdf_delt_cw
    cdf_list = list()
    for method_name, data in cdf_delt_cw.items():
        df = pd.DataFrame(data).transpose()
        df.columns = ['prob_rank', 'cdf_delt_cw']
        cdf_list.append(df)

    plt.figure()
    for method_name, data in cdf_delt_cw.items():
        data = cdf_delt_cw[method_name]
        n_cases = num_cases[method_name]
        x = [0] + data[0] + [n_cases]
        y = [0] + data[1] + [data[1][-1]]
        plt.step(x, y, where='post', label=method_name)
    plt.xlabel("# cases")
    plt.ylabel("delta CW")
    plt.title("Cumulative distribution of delta cost weight (CW_delta)")
    plt.legend()
    # plt.savefig('cdf_delt_cw.png')
    save_figure_to_pdf_on_s3(plt, s3_bucket, os.path.join(dir_output, 'case_ranking_plot_cdf.pdf'))


if __name__ == '__main__':
    create_rankings_of_revised_cases(
        filename_revised_cases="s3://code-scout/performance-measuring/CodeScout_GroundTruthforPerformanceMeasuring.csv",
        filename_codescout_results="s3://code-scout/performance-measuring/case_rankings/DRG_tree/revisions/ksw2019/",
        dir_output="s3://code-scout/performance-measuring/case_rankings/DRG_tree/revisions/ksw2019_case_ranking_tier_plots/",
        s3_bucket='code-scout')
