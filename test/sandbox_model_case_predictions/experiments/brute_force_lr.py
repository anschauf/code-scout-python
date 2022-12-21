import os
import warnings
from os.path import join

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from src import ROOT_DIR
from test.sandbox_model_case_predictions.data_handler import load_data
from test.sandbox_model_case_predictions.utils import get_list_of_all_predictors, list_all_subsets, RANDOM_SEED, \
    prepare_train_eval_test_split, create_performance_app_ground_truth, create_predictions_output_performance_app

hospital_year_for_performance_app = ('KSW', 2020)


dir_output = join(ROOT_DIR, 'results', 'logistic_regression_predictors_screen')
if not os.path.exists(dir_output):
    os.makedirs(dir_output)

revised_case_ids_filename = join(ROOT_DIR, 'resources', 'data', 'revised_case_ids.csv')
revised_cases_in_data = pd.read_csv(revised_case_ids_filename)
create_performance_app_ground_truth(dir_output, revised_cases_in_data,
                                    hospital=hospital_year_for_performance_app[0],
                                    year=hospital_year_for_performance_app[1])

all_data = load_data(only_2_rows=True)
features_dir = join(ROOT_DIR, 'resources', 'features')
feature_filenames, encoders = get_list_of_all_predictors(all_data, features_dir, overwrite=False)
feature_names = sorted(list(feature_filenames.keys()))
n_features = len(feature_names)

ind_X_train, ind_X_test, y_train, y_test, ind_hospital_leave_out, y_hospital_leave_out = \
    prepare_train_eval_test_split(dir_output, revised_cases_in_data,
                                  hospital_leave_out=hospital_year_for_performance_app[0],
                                  year_leave_out=hospital_year_for_performance_app[1])
n_positive_labels_train = int(y_train.sum())


list_model_id = list()
list_model_description = list()
list_f1_measure_train = list()
list_precision_train = list()
list_recall_train = list()
list_accuracy_train = list()
list_f1_measure_test = list()
list_precision_test = list()
list_recall_test = list()
list_accuracy_test = list()
id_counter = 1

for ind_features in list_all_subsets(range(n_features), reverse=True):
    n_features_in_subset = len(ind_features)
    if n_features_in_subset < 1:
        continue

    n_predictors = 0
    for feature_idx in ind_features:
        feature_name = feature_names[feature_idx]

        if feature_name in encoders:
            encoder = encoders[feature_name]
            if isinstance(encoder, MultiLabelBinarizer):
                n_predictors += len(encoder.classes_)
            else:
                n_predictors += sum(len(c) for c in encoder.categories_)

        else:
            n_predictors += 1

    if n_predictors > n_positive_labels_train:
        continue

    X = list()
    features_in_subset = list()
    for feature_idx in ind_features:
        feature_name = feature_names[feature_idx]
        features_in_subset.append(feature_name)
        feature_filename = feature_filenames[feature_name]
        feature_values = np.load(feature_filename, mmap_mode='r', allow_pickle=False, fix_imports=False)
        X.append(feature_values)

    X = np.hstack(X)

    model_description = ', '.join(sorted(features_in_subset))
    logger.info(f'Training model on data [{X.shape[0]} x {X.shape[1]}]; features [{model_description}]')
    list_model_description.append(model_description)

    X_train = X[ind_X_train, :]
    X_test = X[ind_X_test, :]
    X_hospital_left_out = X[ind_hospital_leave_out, :]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        normalizer = StandardScaler()
        X_train = normalizer.fit_transform(X_train)
        X_test = normalizer.transform(X_test)

        # train model
        # model = LogisticRegression(penalty='l1', class_weight='balanced', solver='liblinear', random_state=RANDOM_SEED)
        model = LogisticRegression(class_weight='balanced', random_state=RANDOM_SEED, n_jobs=1)

        model = model.fit(X_train, y_train)

        # predict on train and test
        # noinspection PyUnresolvedReferences
        predictions_train = model.predict_proba(X_train)[:, 1]
        predictions_train_binary = (predictions_train > 0.5).astype(int)

        # noinspection PyUnresolvedReferences
        predictions_test = model.predict_proba(X_test)[:, 1]
        predictions_test_binary = (predictions_test > 0.5).astype(int)

        # compute evaluation metrics
        list_f1_measure_train.append(f1_score(y_train, predictions_train_binary))
        list_precision_train.append(precision_score(y_train, predictions_train_binary))
        list_recall_train.append(recall_score(y_train, predictions_train_binary))
        list_accuracy_train.append(accuracy_score(y_train, predictions_train_binary))

        f1_test = f1_score(y_test, predictions_test_binary)
        precision_test = precision_score(y_test, predictions_test_binary)
        recall_test = recall_score(y_test, predictions_test_binary)
        accuracy_test = accuracy_score(y_test, predictions_test_binary)
        logger.debug(f'{f1_test=:.6f}, {precision_test=:.6f}, {recall_test=:.6f}')

        list_f1_measure_test.append(f1_test)
        list_precision_test.append(precision_test)
        list_recall_test.append(recall_test)
        list_accuracy_test.append(accuracy_test)

        # predict for hospital which was left out
        predictions_hospital_left_out = model.predict_proba(X_hospital_left_out)[:, 1]
        create_predictions_output_performance_app(filename=join(dir_output, f'{str(id_counter)}.csv'),
                                                  case_ids=revised_cases_in_data.iloc[ind_hospital_leave_out]['id'].values,
                                                  predictions=predictions_hospital_left_out)
        list_model_id.append(id_counter)
        id_counter += 1

    # Write results to a file at the end of each iteration
    results = pd.DataFrame({
        'model_id': list_model_id,
        'model_description': list_model_description,
        'precision_test': list_precision_test,
        'recall_test': list_recall_test,
        'f1_test': list_f1_measure_test,
        'accuracy_test': list_accuracy_test,
        'precision_train': list_precision_train,
        'recall_train': list_recall_train,
        'f1_train': list_f1_measure_train,
        'accuracy_train': list_accuracy_train,
    }) \
        .sort_values(by=['precision_test', 'f1_test', 'model_description'], ascending=[False, False, True]) \
        .reset_index()

    results.to_csv(join(dir_output, 'predictors_screen.csv'), index=False)

logger.success('done')
