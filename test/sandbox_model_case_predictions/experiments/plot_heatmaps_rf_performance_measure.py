from os.path import join

from src import ROOT_DIR
from test.sandbox_model_case_predictions.utils import get_screen_summary, plot_heatmap_for_metrics, \
    plot_heatmap_for_metrics_diff_train_minus_test

RESULTS_DIR = join(ROOT_DIR, 'results', 'random_forest_parameter_screen_without_mdc_run_04_KSW_2020')
summary = get_screen_summary(RESULTS_DIR)
plot_heatmap_for_metrics(RESULTS_DIR, summary, 'n_estimator', 'max_depth')
plot_heatmap_for_metrics(RESULTS_DIR, summary, 'min_sample_leaf', 'max_depth')
plot_heatmap_for_metrics_diff_train_minus_test(RESULTS_DIR, summary, 'n_estimator', 'max_depth')
plot_heatmap_for_metrics_diff_train_minus_test(RESULTS_DIR, summary, 'min_sample_leaf', 'max_depth')



