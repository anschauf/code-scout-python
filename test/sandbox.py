import pickle
from os.path import join

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MultiLabelBinarizer

from sandbox_model_case_predictions.data_handler import load_data
from sandbox_model_case_predictions.experiments.random_forest import DISCARDED_FEATURES
from sandbox_model_case_predictions.utils import get_list_of_all_predictors, get_revised_case_ids, \
    prepare_train_eval_test_split
from src import ROOT_DIR
from src.ml.explanation.tree.tree_explainer import TreeExplainer

model_name = 'rf_5000'
# model_name = 'rf_5000_depth5'


all_data = load_data(only_2_rows=True)
features_dir = join(ROOT_DIR, 'resources', 'features')
feature_filenames, encoders = get_list_of_all_predictors(all_data, features_dir, overwrite=False)
feature_names = sorted(list(feature_filenames.keys()))

feature_names = [feature_name for feature_name in feature_names
                 if not any(feature_name.startswith(discarded_feature) for discarded_feature in DISCARDED_FEATURES)]

all_feature_names = list()

for feature_name in feature_names:
    feature_name_wo_suffix = '_'.join(feature_name.split('_')[:-1])

    feature_filename = feature_filenames[feature_name]
    feature_values = np.load(feature_filename, mmap_mode='r', allow_pickle=False, fix_imports=False)

    if feature_name in encoders:
        encoder = encoders[feature_name]
        if isinstance(encoder, MultiLabelBinarizer):
            encoded_names = encoder.classes_

        else:
            if isinstance(encoder.categories, str) and encoder.categories == 'auto':
                encoded_names = encoder.categories_[0]
            else:
                encoded_names = encoder.categories

        all_feature_names.extend(f'{feature_name_wo_suffix}="{encoded_name}"' for encoded_name in encoded_names)

    else:
        all_feature_names.append(feature_name_wo_suffix)

logger.info(f"Loading the model '{model_name}' ...")
with open(join(ROOT_DIR, 'results', 'random_forest_only_true_negatives_KSW_2020', f'{model_name}.pkl'), 'rb') as f:
    model = pickle.load(f)


# -----------------------------------------------------------------------------
# FEATURE IMPORTANCE
# -----------------------------------------------------------------------------
logger.info('Calculating the feature importance ...')

feature_importance = list(zip(all_feature_names, model.feature_importances_))

importances_df = (
    pd.DataFrame(feature_importance, columns=['feature', 'importance'])
    .sort_values(by=['importance', 'feature'], ascending=[False, True])
    .reset_index(drop=True)
)

# print(importances_df[:20])

print(importances_df[~importances_df['feature'].str.startswith('month')][:20])


# -----------------------------------------------------------------------------
# FEATURE CONTRIBUTIONS
# -----------------------------------------------------------------------------
logger.info('Assembling test set ...')
revised_case_ids_filename = join(ROOT_DIR, 'resources', 'data', 'revised_case_ids.csv')
revised_cases_in_data = get_revised_case_ids(all_data, revised_case_ids_filename, overwrite=False)

y = revised_cases_in_data['is_revised'].values
n_samples = y.shape[0]

hospital_year_for_performance_app = ('KSW', 2020)

_, _, _, _, ind_hospital_leave_out, y_hospital_leave_out = \
        prepare_train_eval_test_split(revised_cases_in_data,
                                      hospital_leave_out=hospital_year_for_performance_app[0],
                                      year_leave_out=hospital_year_for_performance_app[1],
                                      only_reviewed_cases=True)


X = list()
for feature_name in feature_names:
    feature_filename = feature_filenames[feature_name]
    feature_values = np.load(feature_filename, mmap_mode='r', allow_pickle=False, fix_imports=False)[ind_hospital_leave_out, :]
    X.append(feature_values)

X_test = np.hstack(X)

logger.info('Calculating the feature contributions ...')

revised_cases_idx = np.where(y_hospital_leave_out == 1)[0]
non_revised_cases_idx = np.where(y_hospital_leave_out == 0)[0]

te = TreeExplainer(model, data=X_test[revised_cases_idx, :], targets=y_hospital_leave_out[revised_cases_idx], n_jobs=-1, verbose=True)
te.n_target_levels = 2
te.feature_names = all_feature_names
te.features_data_types = {feature_name: value for (_, value), feature_name in zip(te.features_data_types.items(), all_feature_names)}
te.explain_feature_contributions(joint_contributions=False, ignore_non_informative_nodes=True)

logger.info(f'Number of correct predictions: {te.correct_predictions.sum()}/{te.correct_predictions.shape[0]}')

# Average the contribution across all the correct predictions for the positive label
feature_contribution_correct = np.mean(te.contributions[te.correct_predictions, :, 1], axis=0)
feature_contribution_incorrect = np.mean(te.contributions[~te.correct_predictions, :, 0], axis=0)

correct_col = 'contribution to correct prediction'
incorrect_col = 'contribution to wrong prediction'
delta_col = 'delta'

df = pd.DataFrame(np.vstack((feature_contribution_correct, feature_contribution_incorrect)).T, index=all_feature_names, columns=[correct_col, incorrect_col])
# df = df[df[correct_col] > 0]
df[delta_col] = df[correct_col] - df[incorrect_col]
df *= 1000
df = df.sort_values(by=delta_col, ascending=False).reset_index()
pd.set_option('display.precision', 5); print(df[:20])


# df2 = pd.DataFrame(te.contributions[0, :, 1].reshape(-1, 1), index=te.feature_names, columns=['revised'])
# df2['feature_value'] = X_test[0, :]
# df2.sort_values(by='revised', ascending=False, inplace=True)
# print(df2[:20])

print('')
