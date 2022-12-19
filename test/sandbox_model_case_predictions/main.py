import os
import shutil
import warnings
from os import makedirs
from os.path import exists, join
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from src import ROOT_DIR
from src.service.bfs_cases_db_service import get_all_revised_cases, get_sociodemographics_by_sociodemographics_ids
from src.service.database import Database
from test.sandbox_model_case_predictions.data_handler import load_data
from test.sandbox_model_case_predictions.utils import get_list_of_all_predictors, list_all_subsets

RANDOM_SEED = 42
OVERWRITE_REVISED_CASE_IDs = False
OVERWRITE_FEATURE_FILES = False


dir_output = join(ROOT_DIR, 'results', 'logistic_regression_predictors_screen')
if not exists(dir_output):
    makedirs(dir_output)

features_dir = join(ROOT_DIR, 'resources', 'features')
if OVERWRITE_FEATURE_FILES:
    shutil.rmtree(features_dir, ignore_errors=True)
Path(features_dir).mkdir(parents=True, exist_ok=True)

revised_case_ids_filename = join(ROOT_DIR, 'resources', 'revised_case_ids.csv')


SHOULD_LOAD_DATA = OVERWRITE_REVISED_CASE_IDs or not os.path.exists(revised_case_ids_filename) or OVERWRITE_FEATURE_FILES

if SHOULD_LOAD_DATA:
    all_data = load_data()
else:
    all_data = load_data(only_2_rows=True)


if OVERWRITE_REVISED_CASE_IDs or not os.path.exists(revised_case_ids_filename):
    with Database() as db:
        revised_cases_all = get_all_revised_cases(db.session)
        revised_case_sociodemographic_ids = revised_cases_all['sociodemographic_id'].values.tolist()
        sociodemographics_revised_cases = get_sociodemographics_by_sociodemographics_ids(revised_case_sociodemographic_ids, db.session)

    revised_cases = sociodemographics_revised_cases[['case_id', 'age_years', 'gender', 'duration_of_stay']].copy()
    revised_cases['revised'] = 1
    logger.info(f'There are {revised_cases.shape[0]} revised cases in the DB')

    revised_cases['case_id'] = revised_cases['case_id'].str.lstrip('0')
    all_data['id'] = all_data['id'].str.lstrip('0')

    revised_cases_in_data = pd.merge(
        revised_cases, all_data[['id', 'AnonymerVerbindungskode', 'ageYears', 'gender', 'durationOfStay', 'hospital']].copy(),
        how='outer',
        left_on=('case_id', 'age_years', 'gender', 'duration_of_stay'), right_on=('id', 'ageYears', 'gender', 'durationOfStay'),
    )

    # Discard the cases which were revised (according to the DB), but are not present in the data we loaded
    revised_cases_in_data = revised_cases_in_data[~revised_cases_in_data['id'].isna()].reset_index(drop=True)
    # Create the "revised" label column, for modeling
    revised_cases_in_data['is_revised'] = (~revised_cases_in_data['revised'].isna()).astype(int)
    revised_cases_in_data = revised_cases_in_data[['id', 'hospital', 'is_revised']]

    num_revised_cases_in_data = int(revised_cases_in_data["is_revised"].sum())
    num_cases = revised_cases_in_data.shape[0]
    logger.info(f'{num_revised_cases_in_data}/{num_cases} ({float(num_revised_cases_in_data) / num_cases * 100:.1f}%) cases were revised')

    revised_cases_in_data.to_csv(revised_case_ids_filename, index=False)

else:
    revised_cases_in_data = pd.read_csv(revised_case_ids_filename)


feature_filenames, encoders = get_list_of_all_predictors(all_data, features_dir, overwrite=OVERWRITE_FEATURE_FILES)

feature_names = sorted(list(feature_filenames.keys()))
n_features = len(feature_names)
logger.info(f'Created {n_features} features')


y = revised_cases_in_data['is_revised'].values
n_samples = y.shape[0]
ind_X_train, ind_X_test, y_train, y_test = train_test_split(range(n_samples), y, stratify=y, test_size=0.3, random_state=RANDOM_SEED)
del all_data

n_positive_labels_train = int(y_train.sum())


list_model_description = list()
list_f1_measure_train = list()
list_precision_train = list()
list_recall_train = list()
list_accuracy_train = list()
list_f1_measure_test = list()
list_precision_test = list()
list_recall_test = list()
list_accuracy_test = list()

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

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        normalizer = StandardScaler()
        X_train = normalizer.fit_transform(X_train)
        X_test = normalizer.transform(X_test)

        # train model
        # model = LogisticRegression(penalty='l1', class_weight='balanced', solver='liblinear', random_state=RANDOM_SEED, n_jobs=-1)
        # model = LogisticRegression(class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1)

        model = RandomForestClassifier(
            n_estimators=1000,
            criterion='entropy',
            n_jobs=-1,
            random_state=RANDOM_SEED,
            class_weight='balanced'
        )

        model = model.fit(X_train, y_train)

        # predict on train and test
        predictions_train = model.predict_proba(X_train)[:, 1]
        predictions_train_binary = (predictions_train > 0.5).astype(int)

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

    # Write results to a file at the end of each iteration
    results = pd.DataFrame({
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

    results.to_csv(join(dir_output, 'predictors_screen.csv'))
