import os.path
from os.path import join

import awswrangler as wr
import pandas as pd

from src import ROOT_DIR
from test.sandbox_model_case_predictions.utils import get_screen_summary_random_forest, S3_PREFIX

s3_bucket: str = 'code-scout'
prefix = f's3://{s3_bucket}'
# prefix = join(ROOT_DIR, 'results')
# dir_results_aucs = f'02_rf_hyperparameter_screen/01_runKSW_2020_results'
for hospital in ['FT_2019', 'HI_2016', 'KSSG_2021', 'KSW_2017', 'KSW_2018', 'KSW_2019', 'KSW_2020', 'LI_2017', 'LI_2018', 'SLI_2019', 'SRRWS_2019']:
    dir_results_aucs = f'brute_force_case_ranking_predictions/02_rf_hyperparameter_screen/predictions_LOO_{hospital}_results/'
    dir_results_local = join(ROOT_DIR, 'results', f'02_rf_hyperparameter_screen/predictions_LOO/{hospital}')
    summary = get_screen_summary_random_forest(dir_results_local)

    filename_aucs = join(prefix, dir_results_aucs, 'area_under_the_curves_top_10.csv')
    if filename_aucs.startswith(S3_PREFIX):
        results_aucs = wr.s3.read_csv(filename_aucs)
    else:
        results_aucs = pd.read_csv(filename_aucs)

    n_estimator = list()
    max_depth = list()
    min_sample_leaf = list()
    min_sample_split = list()
    for name in results_aucs['method'].values:
        split_name = name.split('_')
        n_estimator.append(int(split_name[0].split('-')[-1]))
        max_depth.append(int(split_name[1].split('-')[-1]))
        min_sample_leaf.append(int(split_name[2].split('-')[-1]))
        if len(split_name) > 3:
            min_sample_split.append(int(split_name[3].split('-')[-1]))
        else:
            min_sample_split.append('')

    df_areas_top_10_percent = pd.DataFrame({
        'n_estimator': n_estimator,
        'max_depth': max_depth,
        'min_sample_leaf': min_sample_leaf,
        'min_sample_split': min_sample_split,
        'area': results_aucs['area'].values,
        'area_normalized': results_aucs['area_normalized'].values
    }).sort_values(by='area', ascending=False)

    results_local = pd.read_csv(join(dir_results_local, 'screen_summary.csv'))

    results_merged = pd.merge(df_areas_top_10_percent, results_local, how='outer', on=('n_estimator', 'max_depth', 'min_sample_leaf', 'min_sample_split'))
    dir_output = os.path.dirname(filename_aucs)
    if dir_output.startswith(S3_PREFIX):
        wr.s3.to_csv(results_merged, join(dir_output, 'outer_merged_results.csv'), index=False)
    else:
        results_merged.to_csv(join(dir_output, 'outer_merged_results.csv'), index=False)
