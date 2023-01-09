
import os
import pickle
import sys
import warnings
from os.path import join

import numpy as np
from loguru import logger
from sklearn import clone
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


from src import ROOT_DIR
from test.sandbox_model_case_predictions.data_handler import load_data
from test.sandbox_model_case_predictions.utils import create_predictions_output_performance_app, \
    get_list_of_all_predictors, get_revised_case_ids, RANDOM_SEED

from xgboost import XGBClassifier

RANDOM_FOREST_NUM_TREES = 100
RANDOM_FOREST_MAX_DEPTH = 3
RANDOM_FOREST_MIN_SAMPLES_LEAF = 0.1
LEARNING_RATE = 1

RESULTS_DIR = join(ROOT_DIR, 'results', 'xgboost_only_reviewed', f'n_trees_{RANDOM_FOREST_NUM_TREES}-max_depth_{RANDOM_FOREST_MAX_DEPTH}-learning_rate_{LEARNING_RATE}')
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

REVISED_CASE_IDS_FILENAME = join(ROOT_DIR, 'resources', 'data', 'revised_case_ids.csv')

DISCARDED_FEATURES = ('hospital', 'month_admission', 'month_discharge', 'year_discharge')


def train_random_forest_only_reviewed_cases():
    all_data = load_data(only_2_rows=True)
    features_dir = join(ROOT_DIR, 'resources', 'features')
    feature_filenames, encoders = get_list_of_all_predictors(all_data, features_dir, overwrite=False)
    feature_names = sorted(list(feature_filenames.keys()))

    feature_names = [feature_name for feature_name in feature_names
                     if not any(feature_name.startswith(discarded_feature) for discarded_feature in DISCARDED_FEATURES)]

    revised_cases_in_data = get_revised_case_ids(all_data, REVISED_CASE_IDS_FILENAME, overwrite=False)
    # create_performance_app_ground_truth(dir_output, revised_cases_in_data, hospital_year_for_performance_app[0], hospital_year_for_performance_app[1])

    reviewed_cases = revised_cases_in_data[(revised_cases_in_data['is_revised'] == 1) | (revised_cases_in_data['is_reviewed'] == 1)]
    y = reviewed_cases['is_revised'].values
    sample_indices = reviewed_cases['index'].values

    logger.info('Assembling features ...')
    features = list()

    for feature_name in feature_names:
        feature_filename = feature_filenames[feature_name]
        feature_values = np.load(feature_filename, mmap_mode='r', allow_pickle=False, fix_imports=False)
        features.append(feature_values[sample_indices, :])

    features = np.hstack(features)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        logger.info(f'Training XGBoost with {RANDOM_FOREST_NUM_TREES} trees, max_depth={RANDOM_FOREST_MAX_DEPTH}, learning_rate={LEARNING_RATE} ...')

        # estimator = RandomForestClassifier(
        #     n_estimators=RANDOM_FOREST_NUM_TREES,
        #     max_depth=RANDOM_FOREST_MAX_DEPTH,
        #     min_samples_leaf=RANDOM_FOREST_MIN_SAMPLES_LEAF,
        #     class_weight='balanced',
        #     oob_score=True,
        #     min_impurity_decrease=np.finfo(np.float32).eps,  # the smallest positive number, so that it is not 0
        #     criterion='entropy', n_jobs=-1, random_state=RANDOM_SEED,
        # )

        estimator = XGBClassifier(
            n_estimators=RANDOM_FOREST_NUM_TREES,
            max_depth=RANDOM_FOREST_MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            objective='binary:logistic',
            seed=RANDOM_SEED,
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

        ensemble = list()

        # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        metrics = ['AUROC', 'AUPRC', 'precision', 'recall', 'F1']

        scores = {
            'train_AUROC': list(), 'test_AUROC': list(),
            'train_AUPRC': list(), 'test_AUPRC': list(),
            'train_precision': list(), 'test_precision': list(),
            'train_recall': list(), 'test_recall': list(),
            'train_F1': list(), 'test_F1': list(),
        }

        train_indices_per_model = list()

        for train_indices, test_indices in tqdm(cv.split(features, y), total=5):
            train_indices = np.sort(train_indices)
            test_indices = np.sort(test_indices)

            train_indices_per_model.append(sample_indices[train_indices])

            model = clone(estimator)
            model.fit(features[train_indices, :], y[train_indices])

            ensemble.append(model)
            train_probas = model.predict_proba(features[train_indices, :])
            test_probas = model.predict_proba(features[test_indices, :])

            train_predictions = (train_probas > 0.5)[:, 1].astype(int)
            test_predictions = (test_probas > 0.5)[:, 1].astype(int)

            scores['train_AUROC'].append(roc_auc_score(y[train_indices], train_predictions))
            scores['test_AUROC'].append(roc_auc_score(y[test_indices], test_predictions))

            scores['train_AUPRC'].append(average_precision_score(y[train_indices], train_predictions))
            scores['test_AUPRC'].append(average_precision_score(y[test_indices], test_predictions))

            scores['train_precision'].append(precision_score(y[train_indices], train_predictions))
            scores['test_precision'].append(precision_score(y[test_indices], test_predictions))

            scores['train_recall'].append(recall_score(y[train_indices], train_predictions))
            scores['test_recall'].append(recall_score(y[test_indices], test_predictions))

            scores['train_F1'].append(f1_score(y[train_indices], train_predictions))
            scores['test_F1'].append(f1_score(y[test_indices], test_predictions))

    logger.info('--- Average performance ---')
    performance_log = list()
    performance_log.append(f'# revised: {int(y.sum())}')
    performance_log.append(f'# cases: {y.shape[0]}')

    longest_scorer_name = max(len(name) for name in metrics)
    for metric in metrics:
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
    with open(join(RESULTS_DIR, 'rf_cv.pkl'), 'wb') as f:
        pickle.dump(ensemble, f, fix_imports=False)

    logger.info('Calculating and storing predictions for each combination of hospital and year, which contains revised cases ...')
    # List the hospitals and years for which there are revised cases
    all_hospitals_and_years = revised_cases_in_data[revised_cases_in_data['is_revised'] == 1][['hospital', 'dischargeYear']].drop_duplicates().values.tolist()

    for info in tqdm(all_hospitals_and_years):
        hospital_name = info[0]
        discharge_year = info[1]
        hospital_data = revised_cases_in_data[(revised_cases_in_data['hospital'] == hospital_name) & (revised_cases_in_data['dischargeYear'] == discharge_year)]

        indices = hospital_data['index'].values
        case_ids = hospital_data['id'].values

        test_features = list()
        for feature_name in feature_names:
            feature_filename = feature_filenames[feature_name]
            feature_values = np.load(feature_filename, mmap_mode='r', allow_pickle=False, fix_imports=False)
            test_features.append(feature_values[indices, :])
        test_features = np.hstack(test_features)

        predictions = np.zeros((test_features.shape[0], len(ensemble))) * np.nan
        for idx, model in enumerate(ensemble):
            all_predictions = model.predict_proba(test_features)[:, 1]

            train_indices = train_indices_per_model[idx]
            training_prediction_indices = np.in1d(indices, train_indices, assume_unique=True)
            all_predictions[training_prediction_indices] = np.nan
            predictions[:, idx] = all_predictions

        avg_predictions = np.nanmean(predictions, axis=1)

        create_predictions_output_performance_app(filename=join(RESULTS_DIR, f'predictions_random_forest-{hospital_name}-{discharge_year}-based_on_reviewed_cases.csv'),
                                                  case_ids=case_ids,
                                                  predictions=avg_predictions)

    logger.success('done')


if __name__ == '__main__':
    train_random_forest_only_reviewed_cases()
    sys.exit(0)
