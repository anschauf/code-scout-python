import os
from pathlib import Path

import awswrangler as wr
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from src import ROOT_DIR
from src.files import load_all_rankings, load_revised_cases
from src.schema import case_id_col, prob_most_likely_code_col
from src.utils.general_utils import save_figure_to_pdf_on_s3
from test.sandbox_model_case_predictions.utils import S3_PREFIX


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

    if not dir_output.startswith(S3_PREFIX):
        Path(dir_output).mkdir(parents=True, exist_ok=True)

    rank_col = 'prob_rank'

    top100 = set(np.arange(0, 100))
    top1000 = set(np.arange(0, 1000))

    # Load the revised cases, which are the ground truth for calculating the performance
    revised_cases = load_revised_cases(filename_revised_cases)

    # Load the rankings from CodeScout
    all_rankings = load_all_rankings(dir_rankings)

    cdf_delta_cw = dict()
    num_cases = dict()
    probabilities = dict()
    for hospital_year, method_name, rankings in tqdm(all_rankings):
        # Sort the cases based on probabilities, and add a column indicating the rank
        rankings = rankings.sort_values(by=prob_most_likely_code_col, ascending=False).reset_index(drop=True)
        rankings[rank_col] = rankings[prob_most_likely_code_col].rank(method='max', ascending=False)

        # Perform an inner join between revised cases and ranked results from CodeScout
        revised_cases[case_id_col] = revised_cases['combined_id']
        overlap = pd.merge(revised_cases, rankings, on=case_id_col, how='inner')
        revised_codescout = overlap[[case_id_col, 'CW_old', 'CW_new', rank_col]].sort_values(by=rank_col)

        # add probabilities for revised cases
        probabilities[method_name] = dict()
        probabilities[method_name]['revised'] = overlap[prob_most_likely_code_col].values
        non_revised_cases_overlap = pd.merge(rankings, revised_cases, on=case_id_col, how='left')
        probabilities[method_name]['non_revised'] = non_revised_cases_overlap[prob_most_likely_code_col].values

        # Calculate the delta cost-weight
        revised_codescout['delta_CW'] = revised_codescout['CW_new'].astype(float) - revised_codescout['CW_old'].astype(float)

        # The cumsum is the empirical cumulative distribution function (ECDF)
        revised_codescout['cdf'] = revised_codescout['delta_CW'].cumsum()

        x = revised_codescout[rank_col].values
        y = revised_codescout['cdf'].values

        num_cases[method_name] = rankings.shape[0]
        cdf_delta_cw[method_name] = (x, y)

        # # Computation of the Venn diagram
        # all_cases = set(rankings[rank_col].tolist())
        # revised_codescout_overlap = set(revised_codescout[rank_col].tolist())

        # labels = venn.get_labels([top100, top1000, all_cases, revised_codescout_overlap],
        #                          fill=['number', 'logic'])
        # plt.figure()
        # fig, ax = venn.venn4(labels, names=['Top 100', 'Top 1000', 'All cases', 'Revised cases'])
        # fig.suptitle(f'Case Ranking Tier ({hospital_year})', fontsize=40)
        # if dir_output.startswith(S3_PREFIX):
        #     save_figure_to_pdf_on_s3(fig, s3_bucket, os.path.join(dir_output, 'case_ranking_plot_venn.pdf'))
        # else:
        #     plt.savefig(os.path.join(dir_output, 'case_ranking_plot_venn.pdf'), bbox_inches='tight')
        # plt.close()


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
        x = [0] + list(ranks) + [n_cases]
        y = [0] + list(cdf) + [cdf[-1]]
        plt.step(x, y, where='post', label=method_name)
    plt.xlabel("# cases")
    plt.ylabel("delta CW")
    plt.suptitle("Cumulative distribution of delta cost weight (CW_delta)")
    if len(cdf_delta_cw.items()) < 20:
        plt.legend(loc='best', fancybox=True, framealpha=0.8, bbox_to_anchor=(1.05, 1.05))
    if dir_output.startswith(S3_PREFIX):
        save_figure_to_pdf_on_s3(plt, s3_bucket, os.path.join(dir_output, 'case_ranking_plot_cdf.pdf'))
    else:
        plt.savefig(os.path.join(dir_output, 'case_ranking_plot_cdf.pdf'), bbox_inches='tight')
    plt.close()

    list_areas = list()
    list_areas_top_10_percent = list()
    list_areas_top_10_percent_normalized = list()
    plt.figure()
    for method_name, data in cdf_delta_cw.items():
        ranks, cdf = cdf_delta_cw[method_name]
        n_cases = num_cases[method_name]
        n_cases_top_10_percent = int(n_cases*0.1)
        x = [0] + list(ranks) + [n_cases]
        y = [0] + list(cdf) + [cdf[-1]]

        # add top 10 percent value to be able to compute the AUC under the top 10%
        ind_last_entry_in_top_10_percent = np.max([i for i in range(len(x)) if x[i] < n_cases_top_10_percent])
        y_last_entry_in_top_10_percent = y[ind_last_entry_in_top_10_percent]

        x = x + [n_cases_top_10_percent]
        y = y + [y_last_entry_in_top_10_percent]

        # resort x
        array_sorted = np.sort(np.asarray([x, y]))
        x = list(array_sorted[0, :])
        y = list(array_sorted[1, :])

        area = np.trapz(y, x)
        list_areas.append(area)

        ind_top_10_percent = np.where(np.asarray(x) == float(n_cases_top_10_percent))[0][0] + 1
        area_top_10_percent = np.trapz(y[:ind_top_10_percent], x[:ind_top_10_percent])
        list_areas_top_10_percent.append(area_top_10_percent)
        area_top_10_percent_normalized = np.trapz(np.asarray(y)[:ind_top_10_percent] / np.max(y), np.asarray(x)[:ind_top_10_percent] / n_cases_top_10_percent)
        list_areas_top_10_percent_normalized.append(area_top_10_percent_normalized)

        plt.step(x, y, where='post', label=f'{method_name}_AUC_%.4f' % area_top_10_percent_normalized)
        # plt.step(x, y, where='post', label=f'{method_name}')
    plt.xlabel("# cases")
    plt.ylabel("delta CW")
    n_cases_min = np.min(list(num_cases.values()))
    plt.xlim([0, 0.1*n_cases_min])
    plt.suptitle("Cumulative distribution of delta cost weight (CW_delta)")
    if len(cdf_delta_cw.items()) < 20:
        plt.legend(loc='best', fancybox=True, framealpha=0.8, bbox_to_anchor=(1.05, 1.05))
    if dir_output.startswith(S3_PREFIX):
        save_figure_to_pdf_on_s3(plt, s3_bucket, os.path.join(dir_output, 'case_ranking_plot_cdf_top10_percent.pdf'))
    else:
        plt.savefig(os.path.join(dir_output, 'case_ranking_plot_cdf_top10_percent.pdf'), bbox_inches='tight')
    plt.close()

    df_areas = pd.DataFrame({
        'method': [method_name for method_name, _ in cdf_delta_cw.items()],
        'area': list_areas
    }).sort_values(by='area', ascending=False)
    if dir_output.startswith(S3_PREFIX):
        wr.s3.to_csv(df_areas, os.path.join(dir_output, 'area_under_the_curves.csv'), index=False)
    else:
        df_areas.to_csv(os.path.join(dir_output, 'area_under_the_curves.csv'), index=False)

    df_areas_top_10_percent = pd.DataFrame({
        'method': [method_name for method_name, _ in cdf_delta_cw.items()],
        'area': list_areas_top_10_percent,
        'area_normalized': list_areas_top_10_percent_normalized
    }).sort_values(by='area', ascending=False)
    if dir_output.startswith(S3_PREFIX):
        wr.s3.to_csv(df_areas_top_10_percent, os.path.join(dir_output, 'area_under_the_curves_top_10.csv'), index=False)
    else:
        df_areas_top_10_percent.to_csv(os.path.join(dir_output, 'area_under_the_curves_top_10.csv'), index=False)

    # Cumulative plot for each method from cdf_delta_cw in percent
    plt.figure()
    for method_name, data in cdf_delta_cw.items():
        ranks, cdf = cdf_delta_cw[method_name]
        rank_percent = [cases / num_cases[method_name]*100 for cases in list(ranks)]
        cdf_percent = [cases / max(list(cdf))*100 for cases in list(cdf)]
        x = [0] + rank_percent + [rank_percent[-1]]
        y = [0] + cdf_percent + [cdf_percent[-1]]
        plt.step(x, y, where='post', label=method_name)
    plt.axvline(50, color="red", linestyle="--", linewidth=1)
    plt.xticks(np.linspace(0, 100, num=11))
    plt.yticks(np.linspace(0, 100, num=11))
    plt.xlabel("cases in %")
    plt.ylabel("delta CW in %")
    plt.title("Cumulative distribution of delta cost weight (CW_delta) in %")
    if len(cdf_delta_cw.items()) < 20:
        plt.legend(loc='best', fancybox=True, framealpha=0.8, bbox_to_anchor=(1.05, 1.05))
    if dir_output.startswith(S3_PREFIX):
        save_figure_to_pdf_on_s3(plt, s3_bucket, os.path.join(dir_output, 'case_ranking_plot_cdf_percentage.pdf'))
    else:
        plt.savefig(os.path.join(dir_output, 'case_ranking_plot_cdf_percentage.pdf'), bbox_inches='tight')
    plt.close()

    # plot boxplots to compare probabilities between revised and non-revised cases
    probabilities_dfs = list()
    for method_name, data in probabilities.items():
        probabilities_revised = data['revised']
        probabilities_non_revised = data['non_revised']
        probabilities_dfs.append(pd.DataFrame({
            'Probability': np.concatenate([probabilities_revised, probabilities_non_revised]),
            'Method': np.concatenate([[method_name] * len(probabilities_revised), [method_name] * len(probabilities_non_revised)]),
            'Revision-Outcome': np.concatenate([['Revised'] * len(probabilities_revised), ['Non-revised'] * len(probabilities_non_revised)])
        }))

    plt.figure()
    sns.boxplot(data=pd.concat(probabilities_dfs), x="Probability", y="Method", hue='Revision-Outcome')
    if dir_output.startswith(S3_PREFIX):
        save_figure_to_pdf_on_s3(plt, s3_bucket, os.path.join(dir_output, 'boxplot_probabilities_revised_vs_non-revised.pdf'))
    else:
        plt.savefig(os.path.join(dir_output, 'boxplot_probabilities_revised_vs_non-revised.pdf'), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # folder = '02_rf_hyperparameter_screen/01_runKSW_2020'
    folder = 'global_performance_xgb/test_KSW_2020'

    create_rankings_of_revised_cases(
        filename_revised_cases=os.path.join(ROOT_DIR, f"results/{folder}/ground_truth_performance_app_case_ranking_KSW_2020.csv"),
        dir_rankings=os.path.join(ROOT_DIR, f'results/{folder}/TEST_PREDICTIONS/'),
        dir_output=os.path.join(ROOT_DIR, f"results/{folder}/results"),
        s3_bucket='code-scout'
    )
