import os
import pickle
import sys
import warnings
from os.path import join, exists

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedShuffleSplit
from tqdm import tqdm

from src import ROOT_DIR
from test.sandbox_model_case_predictions.data_handler import load_data
from test.sandbox_model_case_predictions.utils import create_predictions_output_performance_app, \
    get_list_of_all_predictors, get_revised_case_ids, RANDOM_SEED, prepare_train_eval_test_split, \
    create_performance_app_ground_truth


def train_random_forest_only_reviewed_cases():
    for RANDOM_FOREST_NUM_TREES in [1000]:
        # for RANDOM_FOREST_MAX_DEPTH in [2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 10000]:
        # for RANDOM_FOREST_MAX_DEPTH in [2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100, 500, 1000, 5000, 10000]:
        for RANDOM_FOREST_MAX_DEPTH in [9]:
            for RANDOM_FOREST_MIN_SAMPLES_LEAF in [400]:
            # for RANDOM_FOREST_MIN_SAMPLES_LEAF in [1, 2, 3, 4, 5, 10, 100, 200, 300, 400, 500, 1000, 2000]:
                # for RANDOM_FOREST_MIN_SAMPLES_SPLIT in [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 10000]:
                # for RANDOM_FOREST_MIN_SAMPLES_SPLIT in [1, 5, 10, 50, 100, 500, 5000]:
                for RANDOM_FOREST_MIN_SAMPLES_SPLIT in [1]:

                    # for LEAVE_ON_OUT in [
                    #     ('AA', 2017), ('AA', 2018), ('AA', 2019), ('AA', 2020), ('AA', 2021),
                    #     ('AK', 2017), ('AK', 2018), ('AK', 2019), ('AK', 2020),
                    #     ('BC', 2017), ('BC', 2018), ('BC', 2019), ('BC', 2020),
                    #     ('BE', 2017), ('BE', 2018), ('BE', 2019),
                    #     ('BI', 2017), ('BI', 2018), ('BI', 2019), ('BI', 2020), ('BI', 2021),
                    #     ('BS', 2017), ('BS', 2018), ('BS', 2019), ('BS', 2020), ('BS', 2021),
                    #     ('CC', 2017), ('CC', 2018), ('CC', 2019),
                    #     ('FT', 2017), ('FT', 2018), ('FT', 2019),
                    #     ('HI', 2016), ('HI', 2017), ('HI', 2018), ('HI', 2019), ('HI', 2020), ('HI', 2021),
                    #     ('IP', 2017), ('IP', 2018), ('IP', 2019), ('IP', 2020), ('IP', 2021),
                    #     ('KSSG', 2019), ('KSSG', 2020), ('KSSG', 2021),
                    #     ('KSW', 2015), ('KSW', 2017), ('KSW', 2018), ('KSW', 2019), ('KSW', 2020),
                    #     ('LC', 2017), ('LC', 2018), ('LC', 2019),
                    #     ('LI', 2017), ('LI', 2018), ('LI', 2019), ('LI', 2020), ('LI', 2021),
                    #     ('MG', 2017), ('MG', 2018), ('MG', 2019),
                    #     ('PM', 2017), ('PM', 2018), ('PM', 2019),
                    #     ('RO', 2017), ('RO', 2018), ('RO', 2019),
                    #     ('SA', 2015), ('SA', 2016), ('SA', 2017), ('SA', 2018), ('SA', 2019), ('SA', 2020),
                    #     ('SH', 2017), ('SH', 2018), ('SH', 2019), ('SH', 2020),
                    #     ('SLI', 2017), ('SLI', 2018), ('SLI', 2019),
                    #     ('SRRWS', 2017), ('SRRWS', 2018), ('SRRWS', 2019),
                    #     ('ST', 2017), ('ST', 2018), ('ST', 2019), ('ST', 2020),
                    #     ('USZ', 2019)
                    # ]:

                    for LEAVE_ON_OUT in [
                        # ('FT', 2019),
                        # ('HI', 2016),
                        # ('KSSG', 2021),
                        # ('KSW', 2017), ('KSW', 2018), ('KSW', 2019),
                        ('KSW', 2020),
                        # ('LI', 2017), ('LI', 2018),
                        # ('SLI', 2019),
                        # ('SRRWS', 2019)
                    ]:

                        folder_name = f'RF_reviewed_as_positive_label/{LEAVE_ON_OUT[0]}_{LEAVE_ON_OUT[1]}'
                        RESULTS_DIR = join(ROOT_DIR, 'results', folder_name,
                                           f'n_trees_{RANDOM_FOREST_NUM_TREES}-max_depth_{RANDOM_FOREST_MAX_DEPTH}-min_samples_leaf_{RANDOM_FOREST_MIN_SAMPLES_LEAF}-min_samples_split_{RANDOM_FOREST_MIN_SAMPLES_SPLIT}')
                        if not os.path.exists(RESULTS_DIR):
                            os.makedirs(RESULTS_DIR)

                            RESULTS_DIR_TEST_PREDICTIONS = join(ROOT_DIR, 'results', folder_name, 'TEST_PREDICTIONS')
                            if not os.path.exists(RESULTS_DIR_TEST_PREDICTIONS):
                                os.makedirs(RESULTS_DIR_TEST_PREDICTIONS)

                            REVISED_CASE_IDS_FILENAME = join(ROOT_DIR, 'resources', 'data', 'revised_case_ids.csv')

                            DISCARDED_FEATURES = ('hospital', 'month_admission', 'month_discharge', 'year_discharge', 'mdc_OHE', 'hauptkostenstelle_OHE')
                            all_data = load_data(only_2_rows=True)
                            features_dir = join(ROOT_DIR, 'resources', 'features')
                            feature_filenames, encoders = get_list_of_all_predictors(all_data, features_dir, overwrite=False)
                            feature_names = sorted(list(feature_filenames.keys()))

                            feature_names = [feature_name for feature_name in feature_names
                                             if not any(feature_name.startswith(discarded_feature) for discarded_feature in DISCARDED_FEATURES)]

                            revised_cases_in_data = get_revised_case_ids(all_data, REVISED_CASE_IDS_FILENAME, overwrite=False)
                            dir_ground_truth = join(ROOT_DIR, 'results', folder_name)
                            create_performance_app_ground_truth(dir_ground_truth, revised_cases_in_data, LEAVE_ON_OUT[0], LEAVE_ON_OUT[1])
                            # create_performance_app_ground_truth(dir_output, revised_cases_in_data, hospital_year_for_performance_app[0], hospital_year_for_performance_app[1])

                            ind_train, ind_test, _, _, ind_hospital_leave_out, _ = \
                                prepare_train_eval_test_split(dir_output=RESULTS_DIR, revised_cases_in_data=revised_cases_in_data,
                                                              hospital_leave_out=LEAVE_ON_OUT[0],
                                                              year_leave_out=LEAVE_ON_OUT[1],
                                                              only_reviewed_cases=False)

                            # reviewed_cases = revised_cases_in_data[(revised_cases_in_data['is_revised'] == 1) | (revised_cases_in_data['is_reviewed'] == 1)]
                            reviewed_cases = revised_cases_in_data
                            y = reviewed_cases['is_revised'].values + reviewed_cases['is_reviewed'].values
                            y[y > 0] = 1
                            sample_indices = reviewed_cases['index'].values

                            logger.info('Assembling features ...')
                            features = list()
                            feature_ids = list()
                            for feature_name in feature_names:
                                feature_filename = feature_filenames[feature_name]
                                feature_values = np.load(feature_filename, mmap_mode='r', allow_pickle=False, fix_imports=False)
                                features.append(feature_values[sample_indices, :])
                                feature_ids.append([f'{feature_name}_{i}' for i in range(feature_values.shape[1])] if feature_values.shape[1] > 1 else [feature_name])

                            feature_ids = np.concatenate(feature_ids)
                            features = np.hstack(features)

                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore')

                                logger.info(f'Training a RandomForest with {RANDOM_FOREST_NUM_TREES} trees, max_depth={RANDOM_FOREST_MAX_DEPTH}, min_samples_leaf={RANDOM_FOREST_MIN_SAMPLES_LEAF}, min_samples_split={RANDOM_FOREST_MIN_SAMPLES_SPLIT} ...')

                                estimator = RandomForestClassifier(
                                    n_estimators=RANDOM_FOREST_NUM_TREES,
                                    max_depth=RANDOM_FOREST_MAX_DEPTH,
                                    min_samples_leaf=RANDOM_FOREST_MIN_SAMPLES_LEAF,
                                    min_samples_split=RANDOM_FOREST_MIN_SAMPLES_SPLIT,
                                    class_weight='balanced',
                                    oob_score=True,
                                    min_impurity_decrease=np.finfo(np.float32).eps,  # the smallest positive number, so that it is not 0
                                    criterion='entropy', n_jobs=-1, random_state=RANDOM_SEED,
                                )
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
                                scores = cross_validate(estimator, features[ind_train_test], y[ind_train_test], scoring=scoring, cv=cv,
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
                                if hospital_name == LEAVE_ON_OUT[0] and discharge_year == LEAVE_ON_OUT[1]:
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


                                    filename_output = join(RESULTS_DIR_TEST_PREDICTIONS, f'n-estimator-{RANDOM_FOREST_NUM_TREES}_max-depth-{RANDOM_FOREST_MAX_DEPTH}_min-sample-leaf-{RANDOM_FOREST_MIN_SAMPLES_LEAF}_min-samples-split-{RANDOM_FOREST_MIN_SAMPLES_SPLIT}_{LEAVE_ON_OUT[0]}-{LEAVE_ON_OUT[1]}.csv')

                                    create_predictions_output_performance_app(filename=filename_output,
                                                                              case_ids=case_ids,
                                                                              predictions=predictions)

    logger.success('done')


if __name__ == '__main__':
    train_random_forest_only_reviewed_cases()
    sys.exit(0)
