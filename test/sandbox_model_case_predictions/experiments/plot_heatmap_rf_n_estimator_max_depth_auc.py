from os.path import join

import awswrangler as wr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from src.utils.general_utils import save_figure_to_pdf_on_s3

s3_bucket: str = 'code-scout'
prefix = f's3://{s3_bucket}'
dir_results = 'brute_force_case_ranking_predictions/RF_5000/random_forest_parameter_screen_without_mdc_run_05_KSW_2020_plots'
results = wr.s3.read_csv(join(prefix, dir_results, 'area_under_the_curves_top_10.csv'))

n_estimator = list()
max_depth = list()
min_sample_leaf = list()
min_sample_split = list()
auc = list()
for row in results.itertuples():
    parameter_splits = row.method.split('_')
    current_n_estimator = int(parameter_splits[0].split('-')[-1])
    n_estimator.append(current_n_estimator)
    current_max_depth = int(parameter_splits[1].split('-')[-1])
    max_depth.append(current_max_depth)
    current_min_sample_leaf = int(parameter_splits[2].split('-')[-1])
    min_sample_leaf.append(current_min_sample_leaf)
    current_min_sample_split = int(parameter_splits[3].split('-')[-1])
    min_sample_split.append(current_min_sample_split)
    auc.append(row.area_normalized)

summary = pd.DataFrame({
    'n_estimator': n_estimator,
    'max_depth': max_depth,
    'min_sample_leaf': min_sample_leaf,
    'min_sample_split': min_sample_split,
    'AUC': auc
})

def plot_heatmap_to_s3(dir_results, parameter_1, parameter_2):
    combinations = summary[parameter_1].astype('string') + '_' + summary[parameter_2].astype('string')
    if len(np.unique(combinations)) == len(combinations):
        filename = join(dir_results, f'heatmap__{parameter_1}__VS__{parameter_2}__AUC.pdf')
    else:
        filename = join(dir_results, f'MEAN_AGG__heatmap__{parameter_1}__VS__{parameter_2}__AUC.pdf')

    matrix = pd.pivot_table(summary, values='AUC', index=[parameter_1], columns=[parameter_2], aggfunc=np.mean)
    plt.figure()
    sns.heatmap(matrix)
    plt.tight_layout()
    save_figure_to_pdf_on_s3(plt, s3_bucket, filename)
    plt.close()


plot_heatmap_to_s3(dir_results, 'n_estimator', 'max_depth')
plot_heatmap_to_s3(dir_results, 'min_sample_leaf', 'max_depth')
plot_heatmap_to_s3(dir_results, 'min_sample_split', 'max_depth')
plot_heatmap_to_s3(dir_results, 'min_sample_split', 'min_sample_leaf')






