import os
import pickle
import sys
import warnings
from os.path import join

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedShuffleSplit
from tqdm import tqdm

from src import ROOT_DIR
from test.sandbox_model_case_predictions.data_handler import load_data
from test.sandbox_model_case_predictions.utils import create_predictions_output_performance_app, \
    get_list_of_all_predictors, get_revised_case_ids, RANDOM_SEED, prepare_train_eval_test_split

# hand selected raw features
USE_HAND_SELECTED_FEATURES = True
# model selected processed features
USE_MODEL_SELECTED_FEATURES = False
NUM_MODEL_SELECTED_FEATURES = 200

# discarded features when using all features
DISCARDED_FEATURES = (
    'hospital', 'month_admission', 'month_discharge', 'year_discharge', 'mdc_OHE', 'hauptkostenstelle_OHE')

# create folder names for output
LEAVE_ON_OUT = ('KSW', 2020)
# create folder path for storing result with hyperparameter
# Notes: solver and penalty combination
# ‘lbfgs’ - [‘l2’]
# ‘liblinear’ - [‘l1’, ‘l2’]
# ‘newton-cg’ - [‘l2’]
# ‘newton-cholesky’ - [‘l2’]
# ‘sag’ - [‘l2’]
# ‘saga’ - [‘elasticnet’, ‘l1’, ‘l2’]


HYPERPARAMETER = {'C': 1.0, 'penalty': 'l2', 'solver': 'saga'}
hyperparameter_info = '_'.join([f'{key}-{value}' for key, value in HYPERPARAMETER.items()])

if USE_HAND_SELECTED_FEATURES:
    FOLDER_NAME = f'lr_baseline_use_hand_selected_features_{LEAVE_ON_OUT[0]}_{LEAVE_ON_OUT[1]}'
    RESULTS_DIR = join(ROOT_DIR, 'results', FOLDER_NAME,
                       f'{FOLDER_NAME}_{hyperparameter_info}')
    HAND_SELECTED_FEATURES = ('binned_age_RAW', 'drg_cost_weight_RAW', 'mdc_OHE', 'pccl_OHE', 'duration_of_stay_RAW',
                              'gender_OHE', 'number_of_chops_RAW', 'number_of_diags_RAW',
                              'number_of_diags_ccl_greater_0_RAW',
                              'num_drg_relevant_procedures_RAW', 'num_drg_relevant_diagnoses_RAW')

elif USE_MODEL_SELECTED_FEATURES:
    # get top n_num of features selected by random forest
    n_feature = NUM_MODEL_SELECTED_FEATURES
    feature_important_file = 'results/random_forest_parameter_screen_without_mdc_run_06_KSW_2020/n_trees_1000-max_depth_10-min_samples_leaf_20-min_samples_split_200/feature_importances_random_forest.csv'
    feature_importance_df = pd.read_csv(join(ROOT_DIR, feature_important_file))
    feature_important = feature_importance_df['feature'].tolist()
    MODEL_SELECTED_FEATURES = (feature_important[:n_feature])

    FOLDER_NAME = f'lr_baseline_use_model_selected_features_{LEAVE_ON_OUT[0]}_{LEAVE_ON_OUT[1]}'
    RESULTS_DIR = join(ROOT_DIR, 'results', FOLDER_NAME,
                       f'{FOLDER_NAME}_top{n_feature}_{hyperparameter_info}')
#
else:
    FOLDER_NAME = f'lr_baseline_use_all_features_{LEAVE_ON_OUT[0]}_{LEAVE_ON_OUT[1]}'
    RESULTS_DIR = join(ROOT_DIR, 'results', FOLDER_NAME,
                       f'{FOLDER_NAME}_{hyperparameter_info}')

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# predicted results for hospital year for performance app
RESULTS_DIR_TEST_PREDICTIONS = join(ROOT_DIR, 'results', FOLDER_NAME, 'TEST_PREDICTIONS')
if not os.path.exists(RESULTS_DIR_TEST_PREDICTIONS):
    os.makedirs(RESULTS_DIR_TEST_PREDICTIONS)

REVISED_CASE_IDS_FILENAME = join(ROOT_DIR, 'resources', 'data', 'revised_case_ids.csv')


def train_logistic_regression_only_reviewed_cases():
    all_data = load_data(only_2_rows=True)
    features_dir = join(ROOT_DIR, 'resources', 'features')
    feature_filenames, encoders = get_list_of_all_predictors(all_data, features_dir, overwrite=False)
    feature_names = sorted(list(feature_filenames.keys()))
    if USE_HAND_SELECTED_FEATURES:
        feature_names = [feature_name for feature_name in feature_names
                         if any(feature_name.startswith(hand_selected_features) for hand_selected_features in HAND_SELECTED_FEATURES)]
    else:
        feature_names = [feature_name for feature_name in feature_names
                         if not any(feature_name.startswith(discarded_feature) for discarded_feature in DISCARDED_FEATURES)]

    revised_cases_in_data = get_revised_case_ids(all_data, REVISED_CASE_IDS_FILENAME, overwrite=False)
    # create_performance_app_ground_truth(dir_output, revised_cases_in_data, hospital_year_for_performance_app[0], hospital_year_for_performance_app[1])

    ind_train, ind_test, y_train, y_test, ind_hospital_leave_out, y_hospital_leave_out = \
        prepare_train_eval_test_split(revised_cases_in_data=revised_cases_in_data,
                                      hospital_leave_out=LEAVE_ON_OUT[0],
                                      year_leave_out=LEAVE_ON_OUT[1],
                                      only_reviewed_cases=True)

    # reviewed_cases = revised_cases_in_data[(revised_cases_in_data['is_revised'] == 1) | (revised_cases_in_data['is_reviewed'] == 1)]
    reviewed_cases = revised_cases_in_data
    y = reviewed_cases['is_revised'].values
    sample_indices = reviewed_cases['index'].values

    logger.info('Assembling features ...')

    features = list()
    feature_ids = list()
    for feature_name in feature_names:
        feature_filename = feature_filenames[feature_name]
        feature_values = np.load(feature_filename, mmap_mode='r', allow_pickle=False, fix_imports=False)
        features.append(feature_values[sample_indices, :])
        feature_ids.append(
            [f'{feature_name}_{i}' for i in range(feature_values.shape[1])] if feature_values.shape[1] > 1 else [
                feature_name])

    feature_ids = np.concatenate(feature_ids)
    features = np.hstack(features)

    # Get only selected feature_id and feature from other model
    if USE_MODEL_SELECTED_FEATURES:
        selected_feature_idx = np.where(np.isin(feature_ids, MODEL_SELECTED_FEATURES))[0]
        feature_ids = feature_ids[selected_feature_idx]
        features = features[:, selected_feature_idx]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        logger.info(f'Training a LogisticRegression with hyperparameter: {hyperparameter_info} ...')

        estimator = LogisticRegression(C=HYPERPARAMETER['C'], class_weight='balanced',
                                       penalty=HYPERPARAMETER['penalty'], solver=HYPERPARAMETER['solver'],
                                       random_state=RANDOM_SEED)

        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=RANDOM_SEED)

        # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        scoring = {
            'AUROC': 'roc_auc',
            'AUPRC': 'average_precision',
            'precision': 'precision',
            'recall': 'recall',
            'F1': 'f1',
        }

        ind_train_test = np.concatenate([ind_train, ind_test])

        train_features = features[ind_train_test]

        normalizer = StandardScaler()
        train_features_normalized = normalizer.fit_transform(train_features)

        scores = cross_validate(estimator, train_features_normalized, y[ind_train_test], scoring=scoring, cv=cv,
                                return_train_score=True, return_estimator=True, error_score=np.nan,
                                n_jobs=None, verbose=10)

    logger.info('--- Average performance ---')
    performance_log = list()
    performance_log.append(f'# revised: {int(y.sum())}')
    performance_log.append(f'# cases: {y.shape[0]}')

    longest_scorer_name = max(len(name) for name in scoring.keys())
    for metric in scoring.keys():
        train_metric = scores[f'train_{metric}']
        test_metric = scores[f'test_{metric}']
        pad = ' ' * (longest_scorer_name - len(metric) + 1)

        if metric == 'AUROC':
            suffix = ' (random 0.5)'
        elif metric == 'AUPRC':
            random_performance = float(y.sum()) / y.shape[0]
            suffix = f' (random {random_performance:.6f})'
        else:
            suffix = ''

        msg = f'{metric}:{pad}train {np.nanmean(train_metric):.6f}, test {np.nanmean(test_metric):.6f}{suffix}'
        performance_log.append(msg)
        logger.info(msg)

    with open(join(RESULTS_DIR, 'performance.txt'), 'w') as f:
        f.writelines('\n'.join(performance_log))

    logger.info('Storing models ...')
    with open(join(RESULTS_DIR, 'lr_baseline_cv.pkl'), 'wb') as f:
        pickle.dump(scores['estimator'], f, fix_imports=False)

    f_co_eff = list()
    for idx, estimator in enumerate(scores['estimator']):
        f_co_eff.append(np.asarray(estimator.coef_))
    f_co_effs = np.vstack(f_co_eff)
    mean_feature = np.mean(np.vstack(f_co_effs), axis=0)
    std_feature = np.std(np.vstack(f_co_effs), axis=0)

    pd.DataFrame({
        'feature': feature_ids,
        'feature_co_coefficients_mean': mean_feature,
        'feature_co_coefficients_std': std_feature
    }).sort_values(by='feature_co_coefficients_mean', ascending=False).to_csv(
        join(RESULTS_DIR, 'feature_co_coefficients_logistic_regression.csv'), index=False)

    logger.info(
        'Calculating and storing predictions for each combination of hospital and year, which contains revised cases ...')
    # List the hospitals and years for which there are revised cases
    all_hospitals_and_years = revised_cases_in_data[revised_cases_in_data['is_revised'] == 1][
        ['hospital', 'dischargeYear']].drop_duplicates().values.tolist()
    for info in tqdm(all_hospitals_and_years):
        hospital_name = info[0]
        discharge_year = info[1]
        if hospital_name == LEAVE_ON_OUT[0] and discharge_year == LEAVE_ON_OUT[1]:
            hospital_data = revised_cases_in_data[(revised_cases_in_data['hospital'] == hospital_name) & (
                    revised_cases_in_data['dischargeYear'] == discharge_year)]

            indices = hospital_data['index'].values
            case_ids = hospital_data['id'].values

            test_features = list()
            for feature_name in feature_names:
                feature_filename = feature_filenames[feature_name]
                feature_values = np.load(feature_filename, mmap_mode='r', allow_pickle=False, fix_imports=False)
                test_features.append(feature_values[indices, :])
            test_features = np.hstack(test_features)
            test_features_normalized = normalizer.transform(test_features)

            # selected a subset of feature if using Model selected features
            if USE_MODEL_SELECTED_FEATURES:
                selected_feature_idx = np.where(np.isin(feature_ids, MODEL_SELECTED_FEATURES))[0]
                test_features_normalized = test_features_normalized[:, selected_feature_idx]

            predictions = list()
            for model in scores['estimator']:
                predictions.append(model.predict_proba(test_features_normalized)[:, 1])
            predictions = np.mean(np.vstack(predictions), axis=0)

            filename_output = join(RESULTS_DIR_TEST_PREDICTIONS,
                                   f'{FOLDER_NAME}_{hyperparameter_info}.csv')

            create_predictions_output_performance_app(filename=filename_output,
                                                      case_ids=case_ids,
                                                      predictions=predictions)

    logger.success('done')


if __name__ == '__main__':
    train_logistic_regression_only_reviewed_cases()
    sys.exit(0)