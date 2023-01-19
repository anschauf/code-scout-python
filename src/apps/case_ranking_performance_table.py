import os
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
from beartype import beartype
from loguru import logger

from sandbox_model_case_predictions.data_handler import load_data
from sandbox_model_case_predictions.utils import get_revised_case_ids
from src import ROOT_DIR
from src.files import load_all_rankings, load_revised_cases
from src.schema import case_id_col, prob_most_likely_code_col


# noinspection DuplicatedCode
@beartype
def create_summary_for_coders(*,
                              n_cases_to_rank: list[int],
                              prediction_file_prefix: Optional[str] = None
                              ):
    revised_case_ids_filename = os.path.join(ROOT_DIR, 'resources', 'data', 'revised_case_ids.csv')
    all_data = load_data(only_2_rows=True)
    revised_cases_in_data = get_revised_case_ids(all_data, revised_case_ids_filename, overwrite=False)

    n_cases_to_rank = np.array(n_cases_to_rank, dtype=int)
    rank_col = 'prob_rank'

    results_dir = os.path.join(ROOT_DIR, 'results', 'global_performance_xgb')
    tested_datasets = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]

    all_tables = list()

    for tested_dataset in tested_datasets:
        tested_dataset_path = os.path.join(results_dir, tested_dataset)
        files = os.listdir(tested_dataset_path)
        ground_truth_file = [f for f in files if f.startswith('ground_truth')][0]
        ground_truth_file = os.path.join(tested_dataset_path, ground_truth_file)

        try:
            revised_cases = load_revised_cases(ground_truth_file)
        except ValueError:
            logger.error(f'No data for {tested_dataset}')
            continue

        dir_rankings = [f for f in files if f == 'TEST_PREDICTIONS'][0]
        dir_rankings = os.path.join(tested_dataset_path, dir_rankings)
        try:
            rankings = load_all_rankings(dir_rankings)
            if len(rankings) == 1:
                rankings = rankings[0][2]
            elif len(rankings) > 1 and prediction_file_prefix is not None:
                indices = [file_idx
                           for file_idx, f in enumerate(os.listdir(dir_rankings))
                           if f.endswith('.csv') and f.startswith(prediction_file_prefix)]
                if len(indices) == 1:
                    rankings = rankings[indices[0]][2]
                else:
                    raise ValueError(f"Don't know how to select a prediction / ranking file in '{dir_rankings}'")
            else:
                raise ValueError(f"Don't know how to select a prediction / ranking file in '{dir_rankings}'")
        except Exception as e:
            logger.error(e)
            continue

        tested_dataset_info = tested_dataset.removeprefix('test_').replace('_', ' ')
        tested_hospital = tested_dataset_info.split(' ')
        tested_year = int(tested_hospital[1])
        tested_hospital = tested_hospital[0]

        # Sort the cases based on probabilities, and add a column indicating the rank
        rankings = rankings.sort_values(by=prob_most_likely_code_col, ascending=False).reset_index(drop=True)
        rankings[rank_col] = rankings[prob_most_likely_code_col].rank(method='max', ascending=False)

        # Perform an inner join between revised cases and ranked results from CodeScout
        revised_cases[case_id_col] = revised_cases['combined_id']

        revised_cases_dataset = revised_cases_in_data[
            (revised_cases_in_data['hospital'] == tested_hospital) & (revised_cases_in_data['dischargeYear'] == tested_year) &
            ((revised_cases_in_data['is_reviewed'] == 1) | (revised_cases_in_data['is_revised'] == 1))
        ]

        n_true_negatives = np.zeros_like(n_cases_to_rank, dtype=int)
        n_reviewed_cases = revised_cases_dataset['is_reviewed'].sum()

        if n_reviewed_cases > 0:
            reviewed_cases = pd.merge(revised_cases, revised_cases_dataset, left_on=case_id_col, right_on='id', how='outer')
            reviewed_cases = reviewed_cases[['id', 'CW_old', 'CW_new', 'is_revised', 'is_reviewed']]
            reviewed_cases['CW_old'] = reviewed_cases['CW_old'].astype(float).fillna(0.0)
            reviewed_cases['CW_new'] = reviewed_cases['CW_new'].astype(float).fillna(0.0)

            overlap = pd.merge(reviewed_cases, rankings, left_on='id', right_on='CaseId', how='left')
            revised_codescout = overlap[['CW_old', 'CW_new', 'is_revised', 'is_reviewed', rank_col]] \
                .sort_values(by=rank_col) \
                .reset_index(drop=True)

            for idx, n in enumerate(n_cases_to_rank):
                cases_to_consider = revised_codescout[revised_codescout[rank_col] <= n]
                n_true_negatives[idx] = np.where(cases_to_consider['is_reviewed'] == 1)[0].shape[0]

        overlap = pd.merge(revised_cases, rankings, on=case_id_col, how='inner')
        revised_codescout = overlap[[case_id_col, 'CW_old', 'CW_new', rank_col]].sort_values(by=rank_col)

        # Calculate the delta cost-weight
        revised_codescout['delta_CW'] = revised_codescout['CW_new'].astype(float) - revised_codescout['CW_old'].astype(float)

        # The cumsum is the empirical cumulative distribution function (ECDF)
        revised_codescout['cdf'] = revised_codescout['delta_CW'].cumsum()
        total_delta_cw = revised_codescout['delta_CW'].sum()

        ranks = revised_codescout[rank_col].values
        cdf = revised_codescout['cdf'].values
        n_cases = rankings.shape[0]
        n_revised_cases = ranks.shape[0]

        x = np.hstack((ranks, [n_cases])).astype(int)
        y = np.hstack((cdf, [cdf[-1]])).astype(float)

        sum_cost_weights = np.zeros_like(n_cases_to_rank, dtype=float)
        n_true_positives = np.zeros_like(n_cases_to_rank, dtype=int)

        for idx, n in enumerate(n_cases_to_rank):
            cutoff_x = np.where(x < n)[0]
            if cutoff_x.shape[0] == 0:
                sum_cost_weights[idx] = 0
                n_true_positives[idx] = 0

            cutoff_x = cutoff_x[-1]
            sum_cost_weights[idx] = y[cutoff_x]
            n_true_positives[idx] = cutoff_x + 1  # +1 because the index is 0-based

        if n_reviewed_cases == 0:
            n_true_negatives = n_true_positives

        df = pd.DataFrame({
            'dataset': pd.Series(np.tile(tested_dataset_info, n_cases_to_rank.shape), dtype=object),
            'n cases to review': pd.Series(n_cases_to_rank, dtype=int),
            'delta CW': pd.Series(sum_cost_weights, dtype=float),
            'n cases revised': pd.Series(n_true_positives, dtype=int),
            'n cases reviewed (DtoD)': pd.Series(n_true_negatives, dtype=int),
            'delta CW %': pd.Series(sum_cost_weights / total_delta_cw * 100, dtype=float),
            'n cases revised %': pd.Series(n_true_positives / n_revised_cases * 100, dtype=int),
        })

        all_tables.append(df)

    full_performance_table = pd.concat(all_tables, ignore_index=True)

    logger.info('Filtering the table for each n ...')
    for n in n_cases_to_rank:
        performance_at_n = full_performance_table[full_performance_table['n cases to review'] == n]

        performance_at_n = performance_at_n.sort_values(by='delta CW %', ascending=False).reset_index(drop=True)

        precision = (performance_at_n['n cases revised'] / n)
        mean_weighted_precision = np.nansum(precision * performance_at_n['delta CW']) / np.nansum(performance_at_n['delta CW'])

        false_negatives = performance_at_n['n cases reviewed (DtoD)'] - performance_at_n['n cases revised']
        recall = performance_at_n['n cases revised'] / (performance_at_n['n cases revised'] + false_negatives)
        mean_weighted_recall = np.nansum(recall * performance_at_n['delta CW']) / np.nansum(performance_at_n['delta CW'])

        logger.info(f'Average performance @ {n=}: {mean_weighted_precision=}, {mean_weighted_recall=}')

        # Format floating point numbers better
        performance_at_n['delta CW'] = pd.Series([f'{val:.2f}' for val in performance_at_n['delta CW']], index=performance_at_n.index)
        performance_at_n['delta CW %'] = pd.Series([f'{val:.2f}%' for val in performance_at_n['delta CW %']], index=performance_at_n.index)
        performance_at_n['n cases revised %'] = pd.Series([f'{val:.2f}%' for val in performance_at_n['n cases revised %']], index=performance_at_n.index)

        performance_at_n.to_csv(os.path.join(results_dir, f'performance_at_{n}.csv'), index=False)

    logger.success('done')


if __name__ == '__main__':
    prediction_file_prefix = 'xgboost-n_estimators_1000-max_depth_1-eta_001-objective_rank_pairwise_stacked-'

    test_predictions = create_summary_for_coders(
        n_cases_to_rank=[100, 200, 500, 1000],
        prediction_file_prefix=prediction_file_prefix,
    )

    sys.exit(0)
