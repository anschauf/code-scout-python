import os
import pickle
import warnings
from os.path import join

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from src import ROOT_DIR
from test.sandbox_model_case_predictions.data_handler import load_data
from test.sandbox_model_case_predictions.utils import create_predictions_output_performance_app, create_performance_app_ground_truth, \
    get_list_of_all_predictors, prepare_train_eval_test_split, \
    RANDOM_SEED

hospital_year_for_performance_app = ('KSW', 2020)
# hospital_year_for_performance_app = ('LI', 2018)
# hospital_year_for_performance_app = ('SA', 2018)
# hospital_year_for_performance_app = ('KSSG', 2021)

dir_output = join(ROOT_DIR, 'results', f'random_forest_leave_one_feature_out_{hospital_year_for_performance_app[0]}_{hospital_year_for_performance_app[1]}')
if not os.path.exists(dir_output):
    os.makedirs(dir_output)

revised_case_ids_filename = join(ROOT_DIR, 'resources', 'data', 'revised_case_ids.csv')


# noinspection PyUnresolvedReferences
def train_test_random_forest():
    all_data = load_data(only_2_rows=True)
    features_dir = join(ROOT_DIR, 'resources', 'features')
    feature_filenames, encoders = get_list_of_all_predictors(all_data, features_dir, overwrite=False)
    feature_names = sorted(list(feature_filenames.keys()))
    n_features = len(feature_names)

    revised_cases_in_data = pd.read_csv(revised_case_ids_filename)
    create_performance_app_ground_truth(dir_output, revised_cases_in_data, hospital_year_for_performance_app[0], hospital_year_for_performance_app[1])

    ind_train, ind_test, y_train, y_test, ind_hospital_leave_out, y_hospital_leave_out = \
        prepare_train_eval_test_split(revised_cases_in_data,
                                      hospital_leave_out=hospital_year_for_performance_app[0],
                                      year_leave_out=hospital_year_for_performance_app[1],
                                      only_reviewed_cases=True)


    for name in feature_names:
        logger.info(f'Removing feature {name}')
        dir_output_subset = join(dir_output, name)
        if not os.path.exists(dir_output_subset):
            os.makedirs(dir_output_subset)

        feature_names_subset = feature_names.copy()
        feature_names_subset.remove(name)
        n_features = len(feature_names_subset)

        logger.info('Assembling features ...')
        features = list()
        features_in_subset = list()
        features_col_names = list()
        for feature_idx in range(n_features):
            feature_name = feature_names_subset[feature_idx]
            features_in_subset.append(feature_name)
            feature_filename = feature_filenames[feature_name]
            feature_values = np.load(feature_filename, mmap_mode='r', allow_pickle=False, fix_imports=False)
            features_col_names.append([f'{feature_name}_{i}' for i in range(feature_values.shape[1])] if feature_values.shape[1] > 1 else [feature_name])
            features.append(feature_values)

        features = np.hstack(features)

        features_train = features[ind_train, :]
        features_test = features[ind_test, :]
        features_hospital_left_out = features[ind_hospital_leave_out, :]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            logger.info('Training ...')
            # train model
            model = RandomForestClassifier(
                n_estimators=5000,
                max_depth=None,
                criterion='entropy',
                n_jobs=-1,
                random_state=RANDOM_SEED,
                class_weight='balanced'
            )

            model = model.fit(features_train, y_train)

            with open(join(dir_output_subset, 'rf_5000.pkl'), 'wb') as f:
                pickle.dump(model, f, fix_imports=False)

            pd.DataFrame({
                'feature': np.concatenate(features_col_names),
                'feature_importance': model.feature_importances_.flatten()
            }).sort_values(by='feature_importance', ascending=False).to_csv(
                join(dir_output_subset, 'feature_importances_random_forest.csv'), index=False)

            # predict on train and test
            predictions_train = model.predict_proba(features_train)[:, 1]
            predictions_train_binary = (predictions_train > 0.5).astype(int)

            predictions_test = model.predict_proba(features_test)[:, 1]
            predictions_test_binary = (predictions_test > 0.5).astype(int)

            predictions_hospital_left_out = model.predict_proba(features_hospital_left_out)[:, 1]

            # compute evaluation metrics
            f1_train = f1_score(y_train, predictions_train_binary)
            precision_train = precision_score(y_train, predictions_train_binary)
            recall_train = recall_score(y_train, predictions_train_binary)

            f1_test = f1_score(y_test, predictions_test_binary)
            precision_test = precision_score(y_test, predictions_test_binary)
            recall_test = recall_score(y_test, predictions_test_binary)

            create_predictions_output_performance_app(filename=join(dir_output_subset, f'{name}-{hospital_year_for_performance_app[0]}_{hospital_year_for_performance_app[1]}.csv'),
                                                      case_ids=revised_cases_in_data.iloc[ind_hospital_leave_out]['id'].values,
                                                      predictions=predictions_hospital_left_out)

        logger.success(f'{f1_test=:.6f}, {precision_test=:.6f}, {recall_test=:.6f}, {f1_train=:.6f}, {precision_train=:.6f}, {recall_train=:.6f}')


if __name__ == '__main__':
    train_test_random_forest()
