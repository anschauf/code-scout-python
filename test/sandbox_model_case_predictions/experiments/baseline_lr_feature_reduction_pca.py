import os
import pickle
import sys
import warnings
from os.path import join

import matplotlib.pyplot as plt
import numpy as np

from loguru import logger
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from src import ROOT_DIR
from test.sandbox_model_case_predictions.data_handler import load_data
from test.sandbox_model_case_predictions.utils import create_predictions_output_performance_app, \
    get_list_of_all_predictors, get_revised_case_ids, RANDOM_SEED, prepare_train_eval_test_split
import pandas as pd

VARIANCE_PERCENT_PCA = 0.99

# discarded features when using all features
DISCARDED_FEATURES = (
    'hospital', 'month_admission', 'month_discharge', 'year_discharge', 'mdc_OHE', 'hauptkostenstelle_OHE',
    'vectorized_codes')

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
FOLDER_NAME = f'lr_baseline_use_PCA_reduced_features_{LEAVE_ON_OUT[0]}_{LEAVE_ON_OUT[1]}'
RESULTS_DIR = join(ROOT_DIR, 'results', FOLDER_NAME,
                   f'{FOLDER_NAME}_{VARIANCE_PERCENT_PCA * 100}%_varaince_{hyperparameter_info}')

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# predicted results for hospital year for performance app
RESULTS_DIR_TEST_PREDICTIONS = join(ROOT_DIR, 'results', FOLDER_NAME, 'TEST_PREDICTIONS')
if not os.path.exists(RESULTS_DIR_TEST_PREDICTIONS):
    os.makedirs(RESULTS_DIR_TEST_PREDICTIONS)

REVISED_CASE_IDS_FILENAME = join(ROOT_DIR, 'resources', 'data', 'revised_case_ids.csv')


def feature_reduction_only_reviewed_cases():
    all_data = load_data(only_2_rows=True)
    features_dir = join(ROOT_DIR, 'resources', 'features')
    feature_filenames, encoders = get_list_of_all_predictors(all_data, features_dir, overwrite=False)
    feature_names = sorted(list(feature_filenames.keys()))

    feature_names = [feature_name for feature_name in feature_names
                     if not any(
            feature_name.startswith(discarded_feature) for discarded_feature in DISCARDED_FEATURES)]

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

    # PCA using all train data except leave out hospital
    ind_train_test = np.concatenate([ind_train, ind_test])
    features_train = features[ind_train_test]
    y_train = y[ind_train_test]

    # preprocessing features (scaler)
    scaler = MinMaxScaler()
    data_rescaled = scaler.fit_transform(features_train)

    # get the first 2 priciple components for visualization
    pca_2 = PCA(n_components=2)
    train_components = pca_2.fit_transform(data_rescaled)

    train_components_df = pd.DataFrame(train_components, columns=['component1', 'component2'])
    train_components_df['revised'] = y_train
    labels = ['not revised', 'revised']
    colors = ['red', 'green']

    for i, dff in train_components_df.groupby("revised"):
        col = dff.shape[0]*[colors[i]]
        plt.scatter(dff['component1'], dff['component2'], c=col, label= labels[i])

    plt.legend()
    # train_components_df.plot.scatter(x='component1', y='component2',c=colors)
    plt.savefig('pca_2.png')

    # get the first 3 priciple components for visualization
    pca_3 = PCA(n_components=3)
    train_components_3 = pca_3.fit_transform(data_rescaled)

    train_components_3_df = pd.DataFrame(train_components_3, columns=['component1', 'component2', 'component3'])
    train_components_3_df['revised'] = y_train

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    colors_3 = ['red' if i==0 else 'green' for i in y_train]

    for i, dff in train_components_3_df.groupby("revised"):
        col = dff.shape[0] * [colors[i]]
        ax.scatter3D(dff['component1'], dff['component2'],
                     dff['component3'], c=col, label=labels[i])

    plt.legend()
    plt.savefig('pca_3.png')

    # get certain amount of variance
    pca = PCA(VARIANCE_PERCENT_PCA)

    pca.fit(data_rescaled)
    n_com = pca.n_components_
    # get the top number of components
    feature_train_pca = pca.transform(data_rescaled)

    logger.info(f'{n_com} of principle components contribute to {VARIANCE_PERCENT_PCA * 100}% variance')

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

        scores = cross_validate(estimator, feature_train_pca, y_train, scoring=scoring, cv=cv,
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

            # scale test data and extract the top number of components for test set from pca
            data_rescaled_test = scaler.fit_transform(test_features)
            feature_test_from_pca = pca.transform(data_rescaled_test)

            predictions = list()
            for model in scores['estimator']:
                predictions.append(model.predict_proba(feature_test_from_pca)[:, 1])
            predictions = np.mean(np.vstack(predictions), axis=0)

            filename_output = join(RESULTS_DIR_TEST_PREDICTIONS,
                                   f'{FOLDER_NAME}_{VARIANCE_PERCENT_PCA * 100}%_varaince_{hyperparameter_info}.csv')

            create_predictions_output_performance_app(filename=filename_output,
                                                      case_ids=case_ids,
                                                      predictions=predictions)

    logger.success('done')


if __name__ == '__main__':
    feature_reduction_only_reviewed_cases()
    sys.exit(0)
