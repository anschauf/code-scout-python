from os.path import join

import awswrangler as wr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.general_utils import save_figure_to_pdf_on_s3

s3_bucket: str = 'code-scout'
prefix = f's3://{s3_bucket}'
dir_results = 'brute_force_case_ranking_predictions/RF_5000/random_forest_parameter_screen_combined_run05_run06_plots'
summary = wr.s3.read_csv(join(prefix, dir_results, 'outer_merged_results.csv'))

def plot_heatmap_to_s3(dir_results, parameter_1, parameter_2):
    for metric_name in ['precision_train', 'precision_test', 'precision_train-test',
                        'recall_train', 'recall_test', 'recall_train-test',
                        'f1_train', 'f1_test', 'f1_train-test',
                        'area_normalized']:
        combinations = summary[parameter_1].astype('string') + '_' + summary[parameter_2].astype('string')
        if len(np.unique(combinations)) == len(combinations):
            filename = join(dir_results, f'heatmap__{parameter_1}__VS__{parameter_2}__{metric_name}.pdf')
        else:
            filename = join(dir_results, f'MEAN_AGG__heatmap__{parameter_1}__VS__{parameter_2}__{metric_name}.pdf')

        matrix = pd.pivot_table(summary, values=metric_name, index=[parameter_1], columns=[parameter_2], aggfunc=np.mean)
        plt.figure()
        sns.heatmap(matrix)
        plt.tight_layout()
        save_figure_to_pdf_on_s3(plt, s3_bucket, filename)
        plt.close()


def plot_scatter_for_metrics(RESULTS_DIR, parameter, jitter=True, jitter_scale=1):
    for metric_name in ['precision_train', 'precision_test', 'precision_train-test',
                        'recall_train', 'recall_test', 'recall_train-test',
                        'f1_train', 'f1_test', 'f1_train-test',
                        'area_normalized']:
        plt.figure()
        if jitter:
            x = summary[parameter].values + np.random.normal(loc=0, scale=jitter_scale, size=summary[parameter].values.shape)
        else:
            x = summary[parameter].values
        plt.scatter(x, summary[metric_name].values, s=1)
        plt.xlabel(parameter)
        plt.xticks(summary[parameter].values, summary[parameter].values, rotation=90)
        plt.ylabel(metric_name)
        plt.tight_layout()
        save_figure_to_pdf_on_s3(plt, s3_bucket, join(RESULTS_DIR, f'scatter_{parameter}_{metric_name}.pdf'))
        plt.close()

def plot_scatter(RESULTS_DIR, parameter_1, parameter_2, jitter=True, jitter_scale=1):
        plt.figure()
        if jitter:
            x = summary[parameter_1].values + np.random.normal(loc=0, scale=jitter_scale, size=summary[parameter_1].values.shape)
            y = summary[parameter_2].values + np.random.normal(loc=0, scale=jitter_scale, size=summary[parameter_2].values.shape)
        else:
            x = summary[parameter_1].values
            y = summary[parameter_2].values
        plt.scatter(x,y, s=1)
        plt.xlabel(parameter_1)
        # plt.xticks(summary[parameter_1].values, summary[parameter_1].values, rotation=90)
        plt.ylabel(parameter_2)
        # plt.yticks(summary[parameter_2].values, summary[parameter_2].values)
        plt.tight_layout()
        save_figure_to_pdf_on_s3(plt, s3_bucket, join(RESULTS_DIR, f'scatter_{parameter_2}_{parameter_1}.pdf'))
        plt.close()

dir_results_plots = join(dir_results, 'diagnostics')

plot_heatmap_to_s3(dir_results_plots, 'n_estimator', 'max_depth')
plot_heatmap_to_s3(dir_results_plots, 'n_estimator', 'min_sample_leaf')
plot_heatmap_to_s3(dir_results_plots, 'n_estimator', 'min_sample_split')
plot_heatmap_to_s3(dir_results_plots, 'min_sample_leaf', 'max_depth')
plot_heatmap_to_s3(dir_results_plots, 'min_sample_split', 'max_depth')
plot_heatmap_to_s3(dir_results_plots, 'min_sample_split', 'min_sample_leaf')

plot_scatter_for_metrics(dir_results_plots, 'n_estimator')
plot_scatter_for_metrics(dir_results_plots, 'max_depth')
plot_scatter_for_metrics(dir_results_plots, 'min_sample_leaf')
plot_scatter_for_metrics(dir_results_plots, 'min_sample_split')

plot_scatter(dir_results_plots, 'f1_train-test', 'area_normalized', jitter_scale=0.01)
plot_scatter(dir_results_plots, 'precision_train-test', 'area_normalized', jitter_scale=0.01)
plot_scatter(dir_results_plots, 'recall_train-test', 'area_normalized', jitter_scale=0.01)

plot_scatter(dir_results_plots, 'f1_test', 'area_normalized', jitter_scale=0.01)
plot_scatter(dir_results_plots, 'precision_test', 'area_normalized', jitter_scale=0.01)
plot_scatter(dir_results_plots, 'recall_test', 'area_normalized', jitter_scale=0.01)

plot_scatter(dir_results_plots, 'f1_test', 'f1_train-test', jitter_scale=0.01)
plot_scatter(dir_results_plots, 'precision_test', 'precision_train-test', jitter_scale=0.01)
plot_scatter(dir_results_plots, 'recall_test', 'recall_train-test', jitter_scale=0.01)


