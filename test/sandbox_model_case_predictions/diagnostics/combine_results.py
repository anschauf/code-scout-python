from os.path import join

import awswrangler as wr
import pandas as pd

from src import ROOT_DIR
from test.sandbox_model_case_predictions.utils import get_screen_summary

s3_bucket: str = 'code-scout'
prefix = f's3://{s3_bucket}'
dir_results_s3 = 'brute_force_case_ranking_predictions/XG_Boost_3000/xgboost_parameter_screen_test_plots'
dir_results_local = join(ROOT_DIR, 'results', 'xgboost_parameter_screen_test_plots')
summary = get_screen_summary(dir_results_local)

results_s3 = wr.s3.read_csv(join(prefix, dir_results_s3, 'area_under_the_curves_top_10.csv'))
results_local = pd.read_csv(join(dir_results_local, 'screen_summary.csv'))

results_merged = pd.merge(results_s3, results_local, how='outer', on=('n_estimator', 'max_depth', 'min_sample_leaf', 'min_sample_split'))
wr.s3.to_csv(results_merged, join(prefix, dir_results_s3, 'outer_merged_results.csv'), index=False)
