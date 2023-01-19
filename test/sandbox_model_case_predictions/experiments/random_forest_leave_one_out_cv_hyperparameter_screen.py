import os
import pickle
import sys
import warnings
from os.path import join

import numpy as np
import pandas as pd
from loguru import logger
from sklearn import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from sandbox_model_case_predictions.experiments.xgboost_leave_one_out_cv_hyperparameter_screen import score_ranking
from src import ROOT_DIR
from test.sandbox_model_case_predictions.data_handler import load_data
from test.sandbox_model_case_predictions.utils import create_performance_app_ground_truth, \
    create_predictions_output_performance_app, get_list_of_all_predictors, get_revised_case_ids, \
    prepare_train_eval_test_split, RANDOM_SEED


def train_random_forest_only_reviewed_cases():
    REVISED_CASE_IDS_FILENAME = join(ROOT_DIR, 'resources', 'data', 'revised_case_ids.csv')
    all_data = load_data(only_2_rows=True)
    revised_cases_in_data = get_revised_case_ids(all_data, REVISED_CASE_IDS_FILENAME, overwrite=False)

    DISCARDED_FEATURES = ('hospital', 'month_admission', 'month_discharge', 'year_discharge', 'hauptkostenstelle_OHE', 'vectorized_codes')
    # DISCARDED_FEATURES = ('hospital', 'month_admission', 'month_discharge', 'year_discharge')

    for RANDOM_FOREST_NUM_TREES in [1000]:
        # for RANDOM_FOREST_MAX_DEPTH in [2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100]:
        for RANDOM_FOREST_MAX_DEPTH in [10]:
            # for RANDOM_FOREST_MIN_SAMPLES_LEAF in [1, 2, 3, 4, 5, 10, 100, 200, 300, 400, 500, 1000, 2000]:
            for RANDOM_FOREST_MIN_SAMPLES_LEAF in [400]:
                # for RANDOM_FOREST_MIN_SAMPLES_SPLIT in [1, 5, 10, 50, 100, 500, 5000]:
                for RANDOM_FOREST_MIN_SAMPLES_SPLIT in [1]:

                    # all_hospitals_and_years = revised_cases_in_data[revised_cases_in_data['is_revised'] == 1][['hospital', 'dischargeYear']].drop_duplicates().values.tolist()
                    # for LEAVE_ONE_OUT in all_hospitals_and_years:

                    for LEAVE_ONE_OUT in [
                        # ('FT', 2019),
                        # ('HI', 2016),
                        # ('KSSG', 2021),
                        # ('KSW', 2017), ('KSW', 2018), ('KSW', 2019),
                        ('KSW', 2020),
                        # ('LI', 2017), ('LI', 2018),
                        # ('SLI', 2019),
                        # ('SRRWS', 2019),
                        # ('USZ', 2019)
                    ]:
                        folder_name = f'02_rf_hyperparameter_screen/01_run{LEAVE_ONE_OUT[0]}_{LEAVE_ONE_OUT[1]}'
                        filename = f'n_trees_{RANDOM_FOREST_NUM_TREES}-max_depth_{RANDOM_FOREST_MAX_DEPTH}-min_samples_leaf_{RANDOM_FOREST_MIN_SAMPLES_LEAF}-min_samples_split_{RANDOM_FOREST_MIN_SAMPLES_SPLIT}'

                        RESULTS_DIR = join(ROOT_DIR, 'results', folder_name, filename)
                        if True: #not os.path.exists(RESULTS_DIR):
                            os.makedirs(RESULTS_DIR, exist_ok=True)

                            RESULTS_DIR_TEST_PREDICTIONS = join(ROOT_DIR, 'results', folder_name, 'TEST_PREDICTIONS')
                            if True: # not os.path.exists(RESULTS_DIR_TEST_PREDICTIONS):
                                os.makedirs(RESULTS_DIR_TEST_PREDICTIONS, exist_ok=True)

                            features_dir = join(ROOT_DIR, 'resources', 'features')
                            feature_filenames, encoders = get_list_of_all_predictors(all_data, features_dir, overwrite=False)
                            feature_names = sorted(list(feature_filenames.keys()))

                            feature_names = [feature_name for feature_name in feature_names
                                             if not any(feature_name.startswith(discarded_feature) for discarded_feature in DISCARDED_FEATURES)]

                            dir_ground_truth = join(ROOT_DIR, 'results', folder_name)
                            create_performance_app_ground_truth(dir_ground_truth, revised_cases_in_data, LEAVE_ONE_OUT[0], LEAVE_ONE_OUT[1])

                            ind_train, ind_test, y_train, y_test, ind_hospital_leave_out, y_hospital_leave_out = \
                                prepare_train_eval_test_split(dir_output=RESULTS_DIR, revised_cases_in_data=revised_cases_in_data,
                                                              hospital_leave_out=LEAVE_ONE_OUT[0],
                                                              year_leave_out=LEAVE_ONE_OUT[1],
                                                              only_reviewed_cases=True)

                            # reviewed_cases = revised_cases_in_data[(revised_cases_in_data['is_revised'] == 1) | (revised_cases_in_data['is_reviewed'] == 1)]
                            ind_train_test = np.sort(np.hstack((ind_train, ind_test)))
                            reviewed_cases = revised_cases_in_data[np.in1d(revised_cases_in_data['index'], ind_train_test)]
                            y = reviewed_cases['is_revised'].values
                            sample_indices = reviewed_cases['index'].values

                            logger.info('Assembling training features ...')
                            features = list()
                            feature_ids = list()
                            for feature_name in feature_names:
                                feature_filename = feature_filenames[feature_name]
                                feature_values = np.load(feature_filename, mmap_mode='r', allow_pickle=False, fix_imports=False)
                                features.append(feature_values[sample_indices, :])
                                feature_ids.append([f'{feature_name}_{i}' for i in range(feature_values.shape[1])] if feature_values.shape[1] > 1 else [feature_name])

                            feature_ids = np.concatenate(feature_ids)
                            features = np.hstack(features)

                            all_reviewed_case_indices = revised_cases_in_data[(revised_cases_in_data['is_revised'] == 1) | (revised_cases_in_data['is_reviewed'] == 1)]['index'].values
                            ind_hospital_leave_out_reviewed_ind = np.in1d(ind_hospital_leave_out, all_reviewed_case_indices)
                            ind_hospital_leave_out_reviewed = ind_hospital_leave_out[ind_hospital_leave_out_reviewed_ind]
                            y_hospital_leave_out_reviewed = y_hospital_leave_out[ind_hospital_leave_out_reviewed_ind]

                            logger.info('Assembling test features ...')
                            test_features_all = list()
                            test_features_reviewed = list()
                            for feature_name in feature_names:
                                feature_filename = feature_filenames[feature_name]
                                feature_values = np.load(feature_filename, mmap_mode='r', allow_pickle=False, fix_imports=False)
                                test_features_all.append(feature_values[ind_hospital_leave_out, :])
                                test_features_reviewed.append(feature_values[ind_hospital_leave_out_reviewed, :])

                            test_features_all = np.hstack(test_features_all)
                            test_features_reviewed = np.hstack(test_features_reviewed)

                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore')

                                logger.info(f'Training a RandomForest with {RANDOM_FOREST_NUM_TREES} trees, max_depth={RANDOM_FOREST_MAX_DEPTH}, min_samples_leaf={RANDOM_FOREST_MIN_SAMPLES_LEAF}, min_samples_split={RANDOM_FOREST_MIN_SAMPLES_SPLIT} ...')

                                estimator = RandomForestClassifier(
                                    n_estimators=RANDOM_FOREST_NUM_TREES,
                                    max_depth=RANDOM_FOREST_MAX_DEPTH,
                                    min_samples_leaf=RANDOM_FOREST_MIN_SAMPLES_LEAF,
                                    min_samples_split=RANDOM_FOREST_MIN_SAMPLES_SPLIT,
                                    class_weight='balanced',
                                    oob_score=False,
                                    min_impurity_decrease=np.finfo(np.float32).eps,  # the smallest positive number, so that it is not 0
                                    criterion='entropy', n_jobs=-1, random_state=RANDOM_SEED,
                                )
                                cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=RANDOM_SEED)
                                n_models = cv.n_splits

                                # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
                                scoring = {
                                    'AUROC': 'roc_auc',
                                    'AUPRC': 'average_precision',
                                    'precision': 'precision',
                                    'recall': 'recall',
                                    'F1': 'f1',
                                }

                                scores = dict()
                                scores['estimator'] = list()
                                scores['test_predictions'] = np.zeros((test_features_all.shape[0], n_models), dtype=np.float32)
                                scores['train_ndcg'] = np.zeros(n_models, dtype=np.float32)
                                scores['eval_ndcg'] = np.zeros_like(scores['train_ndcg'])
                                scores['test_ndcg'] = np.zeros_like(scores['train_ndcg'])
                                scores['test_ndcg_reviewed'] = np.zeros_like(scores['test_ndcg'])

                                for split_idx, (ind_train, ind_eval) in enumerate(tqdm(cv.split(features, y), total=n_models)):
                                    ind_train = np.sort(ind_train)
                                    ind_eval = np.sort(ind_eval)
                                    train_labels = y[ind_train]

                                    labels, sample_weights = np.unique(y[ind_train], return_counts=True)
                                    sample_weights = sample_weights.astype(float) / sample_weights.sum()
                                    balanced_sample_weights = {l: 1 - sw for l, sw in zip(labels, sample_weights)}
                                    sample_weights = np.array([balanced_sample_weights[l] for l in y[ind_train]], dtype=float)

                                    rf_model = clone(estimator)
                                    rf_model.fit(features[ind_train, :], y[ind_train], sample_weight=sample_weights)

                                    leaf_index = rf_model.apply(features[ind_train, :])
                                    lr_model = LogisticRegression(penalty='l1', class_weight='balanced', solver='liblinear', random_state=RANDOM_SEED)
                                    lr_model.fit(leaf_index, y[ind_train], sample_weight=sample_weights)

                                    scores['estimator'].append((rf_model, lr_model))

                                    def predict(features_mat):
                                        eval_leaf_index = rf_model.apply(features_mat)
                                        probas = lr_model.predict_proba(eval_leaf_index)
                                        return probas[:, 1]

                                    train_predictions = predict(features[ind_train, :])
                                    train_scores = score_ranking(train_predictions, train_labels, np.zeros_like(train_labels), 1)
                                    scores['train_ndcg'][split_idx] = train_scores[0]

                                    eval_predictions = predict(features[ind_eval, :])
                                    eval_scores = score_ranking(eval_predictions, y[ind_eval], np.zeros_like(ind_eval), 1)
                                    scores['eval_ndcg'][split_idx] = eval_scores[0]

                                    test_predictions_all = predict(test_features_all)
                                    test_scores_all = score_ranking(test_predictions_all, y_hospital_leave_out, np.zeros_like(y_hospital_leave_out, dtype=int), 1)
                                    scores['test_ndcg'][split_idx] = test_scores_all[0]
                                    scores['test_predictions'][:, split_idx] = test_predictions_all  # Will be stored for the "case ranking" app

                                    test_predictions_reviewed = predict(test_features_reviewed)
                                    test_scores_reviewed = score_ranking(test_predictions_reviewed,
                                                                         y_hospital_leave_out_reviewed,
                                                                         np.zeros_like(y_hospital_leave_out_reviewed,
                                                                                       dtype=int), 1)
                                    scores['test_ndcg_reviewed'][split_idx] = test_scores_reviewed[0]

                                # scores = cross_validate(estimator, features, y, scoring=scoring, cv=cv,
                                #                         return_train_score=True, return_estimator=True, error_score=np.nan,
                                #                         n_jobs=None, verbose=10)

                                logger.info('--- Average performance ---')
                                performance_log = list()
                                performance_log.append(f'# revised: {int(y.sum())}')
                                performance_log.append(f'# cases: {y.shape[0]}')

                                longest_scorer_name = max(len(name) for name in scoring.keys())
                                for metric_name, sklearn_metric in scoring.items():
                                    # Calculate and store the performance on the test set (the left-out hospital)
                                    scorer = get_scorer(sklearn_metric)

                                    test_metric_all = np.zeros(n_models, dtype=np.float32)
                                    test_metric_reviewed = np.zeros(n_models, dtype=np.float32)
                                    for idx, model in enumerate(scores['estimator']):
                                        try:
                                            test_metric_all[idx] = scorer(model, test_features_all, y_hospital_leave_out)
                                        except ValueError:
                                            pass

                                        try:
                                            test_metric_reviewed[idx] = scorer(model, test_features_reviewed, y_hospital_leave_out_reviewed)
                                        except ValueError:
                                            pass

                                    train_metric = scores[f'train_{metric_name}']
                                    eval_metric = scores[f'test_{metric_name}']
                                    pad = ' ' * (longest_scorer_name - len(metric_name) + 1)

                                    if metric_name == 'AUROC':
                                        suffix = ' (random 0.5)'
                                    elif metric_name == 'AUPRC':
                                        random_performance = float(y.sum()) / y.shape[0]
                                        suffix = f' (random {random_performance:.6f})'
                                    else:
                                        suffix = ''

                                    msg = f'{metric_name}:{pad}train {np.nanmean(train_metric):.6f}, eval {np.nanmean(eval_metric):.6f}, test (all) {np.nanmean(test_metric_all):.6f}, test (reviewed) {np.nanmean(test_metric_reviewed):.6f}{suffix}'
                                    performance_log.append(msg)
                                    logger.info(msg)

                            with open(join(RESULTS_DIR, 'performance.txt'), 'w') as f:
                                f.writelines('\n'.join(performance_log))

                            logger.info('Storing models ...')
                            with open(join(RESULTS_DIR, 'rf_cv.pkl'), 'wb') as f:
                                pickle.dump(scores['estimator'], f, fix_imports=False)

                                f_importances = list()
                                for idx, estimator in enumerate(scores['estimator']):
                                    f_importances.append(np.asarray(estimator.feature_importances_))
                                f_importances = np.vstack(f_importances)
                                mean_feature = np.mean(np.vstack(f_importances), axis=0)
                                std_feature = np.std(np.vstack(f_importances), axis=0)

                                pd.DataFrame({
                                    'feature': feature_ids,
                                    'feature_importance_mean': mean_feature,
                                    'feature_importance_std': std_feature
                                }).sort_values(by='feature_importance_mean', ascending=False).to_csv(
                                    join(RESULTS_DIR, 'feature_importances_random_forest.csv'), index=False)

                            logger.info('Calculating and storing predictions for each combination of hospital and year, which contains revised cases ...')
                            # List the hospitals and years for which there are revised cases
                            all_hospitals_and_years = revised_cases_in_data[revised_cases_in_data['is_revised'] == 1][['hospital', 'dischargeYear']].drop_duplicates().values.tolist()
                            for info in tqdm(all_hospitals_and_years):
                                hospital_name = info[0]
                                discharge_year = info[1]
                                if hospital_name == LEAVE_ONE_OUT[0] and discharge_year == LEAVE_ONE_OUT[1]:
                                    hospital_data = revised_cases_in_data[(revised_cases_in_data['hospital'] == hospital_name) & (revised_cases_in_data['dischargeYear'] == discharge_year)]

                                    indices = hospital_data['index'].values
                                    case_ids = hospital_data['id'].values

                                    test_features = list()
                                    for feature_name in feature_names:
                                        feature_filename = feature_filenames[feature_name]
                                        feature_values = np.load(feature_filename, mmap_mode='r', allow_pickle=False, fix_imports=False)
                                        test_features.append(feature_values[indices, :])
                                    test_features = np.hstack(test_features)

                                    predictions = list()
                                    for model in scores['estimator']:
                                        predictions.append(model.predict_proba(test_features)[:, 1])
                                    predictions = np.mean(np.vstack(predictions), axis=0)


                                    filename_output = join(RESULTS_DIR_TEST_PREDICTIONS, f'{filename}-{LEAVE_ONE_OUT[0]}-{LEAVE_ONE_OUT[1]}.csv')

                                    create_predictions_output_performance_app(filename=filename_output,
                                                                              case_ids=case_ids,
                                                                              predictions=predictions)

    logger.success('done')


if __name__ == '__main__':
    train_random_forest_only_reviewed_cases()
    sys.exit(0)
