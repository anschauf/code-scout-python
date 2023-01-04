from os import listdir
from os.path import join

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src import ROOT_DIR
from src.utils.general_utils import save_figure_to_pdf_on_s3

RESULTS_DIR = join(ROOT_DIR, 'results', 'random_forest_parameter_screen_without_mdc_run_02_KSW_2020')
all_runs = listdir(RESULTS_DIR)
all_runs = [f for f in all_runs if f.startswith('n_trees')]

n_estimator = list()
max_depth = list()
precision_train = list()
precision_test = list()
recall_train = list()
recall_test = list()
f1_train = list()
f1_test = list()

for file in all_runs:
    n_estimator.append(int(file.split('-')[0].split('_')[-1]))
    max_depth.append(int(file.split('-')[1].split('_')[-1]))

    with open(join(RESULTS_DIR, file, 'performance.txt')) as f:
        lines = f.readlines()

    precision_train.append(float(lines[4].split(',')[0].split(' ')[-1]))
    precision_test.append(float(lines[4].split(',')[1].split(' ')[-1]))

    recall_train.append(float(lines[5].split(',')[0].split(' ')[-1]))
    recall_test.append(float(lines[5].split(',')[1].split(' ')[-1]))

    f1_train.append(float(lines[6].split(',')[0].split(' ')[-1]))
    f1_test.append(float(lines[6].split(',')[1].split(' ')[-1]))


# plot simple metric
for metric, metric_name in [(precision_train, 'precision_train'),
                            (precision_test, 'precision_test'),
                            (recall_train, 'recall_train'),
                            (recall_test, 'recall_test'),
                            (f1_train, 'f1_train'),
                            (f1_test, 'f1_test')]:
    df_result = pd.DataFrame({
        'n_estimator': n_estimator,
        'max_depth': max_depth,
        metric_name: metric
    })

    plt.figure()
    sns.heatmap(df_result.pivot("n_estimator", "max_depth", metric_name))
    plt.savefig(join(RESULTS_DIR, f'heatmap_{metric_name}.pdf'))
    plt.close()

# plot difference between train and test
for metric, metric_name in [(np.asarray(precision_train) - np.asarray(precision_test), 'precision_train_minus_test'),
                            (np.asarray(recall_train) - np.asarray(recall_test), 'recall_train_minus_test'),
                            (np.asarray(f1_train) - np.asarray(f1_test), 'f1_train_minus_test')]:
    df_result = pd.DataFrame({
        'n_estimator': n_estimator,
        'max_depth': max_depth,
        metric_name: metric
    })

    plt.figure()
    sns.heatmap(df_result.pivot("n_estimator", "max_depth", metric_name))
    plt.savefig(join(RESULTS_DIR, f'heatmap_{metric_name}.pdf'))
    plt.close()



