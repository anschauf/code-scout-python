from os.path import join

import awswrangler as wr
import pandas as pd

from src import ROOT_DIR
from test.sandbox_model_case_predictions.utils import get_screen_summary_random_forest

s3_bucket: str = 'code-scout'
prefix = f's3://{s3_bucket}'
dir_results_s3 = f'brute_force_case_ranking_predictions/RF_5000/random_forest_optimal_models_KSW_2020_plots'
dir_results_local = join(ROOT_DIR, 'results', f'random_forest_optimal_models_KSW_2020')
summary = get_screen_summary_random_forest(dir_results_local)

results_s3 = wr.s3.read_csv(join(prefix, dir_results_s3, 'area_under_the_curves_top_10.csv'))

n_estimator = list()
max_depth = list()
min_sample_leaf = list()
min_sample_split = list()
for name in results_s3['method'].values:
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
    'area': results_s3['area'].values,
    'area_normalized': results_s3['area_normalized'].values
}).sort_values(by='area', ascending=False)

results_local = pd.read_csv(join(dir_results_local, 'screen_summary.csv'))

results_merged = pd.merge(df_areas_top_10_percent, results_local, how='outer', on=('n_estimator', 'max_depth', 'min_sample_leaf', 'min_sample_split'))
wr.s3.to_csv(results_merged, join(prefix, dir_results_s3, 'outer_merged_results.csv'), index=False)
