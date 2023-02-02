import os
import pickle
import sys
import warnings
from os.path import join
import numpy as np
import pandas as pd
from loguru import logger

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from pygam import LogisticGAM, s, f

from src import ROOT_DIR
from test.sandbox_model_case_predictions.data_handler import load_data
from test.sandbox_model_case_predictions.utils import create_predictions_output_performance_app, \
    get_list_of_all_predictors, get_revised_case_ids, RANDOM_SEED, prepare_train_eval_test_split
from tqdm import tqdm


# discarded features when using all features
DISCARDED_FEATURES = (
    'hospital', 'month_admission', 'month_discharge', 'year_discharge', 'mdc_OHE', 'hauptkostenstelle_OHE', 'vectorized_codes')

# hand selected raw features
USE_HAND_SELECTED_FEATURES = True

# create folder names for output
LEAVE_ON_OUT = ('KSW', 2020)

TRAINING_SET_LIMIT = 100000 # -1 for no limit
if USE_HAND_SELECTED_FEATURES:
    FOLDER_NAME = f'GAM_hand_selected_default_settings_{LEAVE_ON_OUT[0]}_{LEAVE_ON_OUT[1]}'
    HAND_SELECTED_FEATURES = ('binned_age_RAW', 'drg_cost_weight_RAW', 'mdc_OHE', 'pccl_OHE', 'duration_of_stay_RAW',
                              'gender_OHE', 'number_of_chops_RAW', 'number_of_diags_RAW',
                              'number_of_diags_ccl_greater_0_RAW', 'num_drg_relevant_procedures_RAW', 'num_drg_relevant_diagnoses_RAW')
else:
    FOLDER_NAME = f'GAM_hand_selected_default_settings_{LEAVE_ON_OUT[0]}_{LEAVE_ON_OUT[1]}'

# predicted results for hospital year for performance app
RESULTS_DIR_TEST_PREDICTIONS = join(ROOT_DIR, 'results', FOLDER_NAME, 'TEST_PREDICTIONS')
if not os.path.exists(RESULTS_DIR_TEST_PREDICTIONS):
    os.makedirs(RESULTS_DIR_TEST_PREDICTIONS)

REVISED_CASE_IDS_FILENAME = join(ROOT_DIR, 'resources', 'data', 'revised_case_ids.csv')

def LogisticGAM_hand_selected_features_only_reviewed_cases():
    all_data = load_data(only_2_rows=True)
    features_dir = join(ROOT_DIR, 'resources', 'features')
    feature_filenames, encoders = get_list_of_all_predictors(all_data, features_dir, overwrite=False)
    feature_names = sorted(list(feature_filenames.keys()))

    if USE_HAND_SELECTED_FEATURES:
        feature_names = [feature_name for feature_name in feature_names
                         if any(
                feature_name.startswith(hand_selected_features) for hand_selected_features in HAND_SELECTED_FEATURES)]
    else:
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

    ind_all = np.concatenate([ind_train, ind_test])
    reviewed_cases_new = reviewed_cases.iloc[ind_all]
    df = reviewed_cases_new[(reviewed_cases_new['hospital'] == 'KSW') & (
            reviewed_cases_new['dischargeYear'] == 2020)]

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

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        logger.info(f'Training a LogisticGAM with gridsearch:  ...')

        # Run LogisticGAM with default settings

        LogGAM = LogisticGAM().gridsearch(features_train, y_train, objective='auto',
                                                                        return_scores=True)

        RESULTS_DIR = join(ROOT_DIR, 'results', FOLDER_NAME,
                           f'{FOLDER_NAME}')

        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)

        # Generating results for training set

        predictions_train = list()
        for model in LogGAM:
            predictions_train.append(model.predict_proba(features_train))

        threshold = y_train.sum()/len(y_train)

        predictions_train_binary = list()
        for values in predictions_train:
            predictions_train_binary.append((values > threshold).astype(int))

        # Generate Performance Metrics of training set

        f1_train = list()
        for binary in predictions_train_binary:
            f1_train.append(f1_score(y_train, binary))

        recall_train = list()
        for binary in predictions_train_binary:
            recall_train.append(recall_score(y_train, binary))

        precision_train = list()
        for binary in predictions_train_binary:
            precision_train.append(precision_score(y_train, binary))

        accuracy_train = list()
        for binary in predictions_train_binary:
            accuracy_train.append(accuracy_score(y_train, binary))

        # Generating results for test set

        predictions_test = list()
        for model in LogGAM:
            predictions_test.append(model.predict_proba(features_test))

        predictions_test_binary = list()
        for values in predictions_test:
            predictions_test_binary.append((values > threshold).astype(int))

        f1_test = list()
        for binary in predictions_test_binary:
            f1_test.append(f1_score(y_test, binary))

        recall_test = list()
        for binary in predictions_test_binary:
            recall_test.append(recall_score(y_test, binary))

        precision_test = list()
        for binary in predictions_test_binary:
            precision_test.append(precision_score(y_test, binary))

        accuracy_test = list()
        for binary in predictions_test_binary:
            accuracy_test.append(accuracy_score(y_test, binary))

        # Write metrics to a file

        metrics = pd.DataFrame({
            'precision_train': precision_train,
            'recall_train': recall_train,
            'f1_train': f1_train,
            'accuracy_train': accuracy_train,
            'precision_test': precision_test,
            'recall_test': recall_test,
            'f1_test': f1_test,
            'accuracy_test': accuracy_test
        }) \
                .sort_values(by=['precision_test', 'f1_test'], ascending=[False, False]) \
            .reset_index()

        metrics.to_csv(join(RESULTS_DIR, f'GAM_predictors_screen_default.csv'))


        #logger.info('Calculating and storing predictions for each combination of hospital and year, which contains revised cases ...')

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

                predictions = list()

                for model in LogGAM:
                    predictions.append(model.predict_proba(test_features))
                predictions = np.mean(np.vstack(predictions), axis=0)

                filename_output = join(RESULTS_DIR_TEST_PREDICTIONS,
                                       f'{FOLDER_NAME}.csv')

                create_predictions_output_performance_app(filename=filename_output,
                                                       case_ids=case_ids,
                                                       predictions=predictions)

    logger.success('done')

if __name__ == '__main__':
    LogisticGAM_hand_selected_features_only_reviewed_cases()
    sys.exit(0)
