from os.path import join

import awswrangler as wr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.general_utils import save_figure_to_pdf_on_s3

s3_bucket: str = 'code-scout'
prefix = f's3://{s3_bucket}'
dir_results = 'brute_force_case_ranking_predictions/RF_5000/random_forest_parameter_screen_without_mdc_run_02_KSW_2020_plots'
results = wr.s3.read_csv(join(prefix, dir_results, 'area_under_the_curves_top_10.csv'))

n_estimator = list()
max_depth = list()
auc = list()
for row in results.itertuples():
    parameter_splits = row.method.split('_')
    current_n_estimator = int(parameter_splits[0].split('-')[-1])
    n_estimator.append(current_n_estimator)
    current_max_depth = int(parameter_splits[1].split('-')[-1])
    max_depth.append(current_max_depth)
    auc.append(row.area_normalized)

df_results = pd.DataFrame({
    'n_estimator': n_estimator,
    'max_depth': max_depth,
    'AUC': auc
})

plt.figure()
sns.heatmap(df_results.pivot("n_estimator", "max_depth", "AUC"))
save_figure_to_pdf_on_s3(plt, s3_bucket, join(dir_results, 'heatmap_n-estimator_max-depth_AUC.pdf'))
plt.close()







