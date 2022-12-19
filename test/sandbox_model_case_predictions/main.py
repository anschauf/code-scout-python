import shutil
from os import makedirs
from os.path import exists, join
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src import ROOT_DIR
from test.sandbox_model_case_predictions.data_handler import load_data
from test.sandbox_model_case_predictions.utils import get_list_of_all_predictors, list_all_subsets

RANDOM_SEED = 42
OVERWRITE_FEATURE_FILES = False


dir_output = join(ROOT_DIR, 'results', 'logistic_regression_predictors_screen')
if not exists(dir_output):
    makedirs(dir_output)

features_dir = join(ROOT_DIR, 'resources', 'features')
if OVERWRITE_FEATURE_FILES:
    shutil.rmtree(features_dir, ignore_errors=True)
Path(features_dir).mkdir(parents=True, exist_ok=True)

all_data = load_data()

feature_filenames, encoders = get_list_of_all_predictors(all_data, features_dir, overwrite=OVERWRITE_FEATURE_FILES)

feature_names = sorted(list(feature_filenames.keys()))
n_features = len(feature_names)
logger.info(f'Created {n_features} features')


y = np.random.randint(0, 2, size=(all_data.shape[0], )) # TODO define correct labels from DB
n_samples = y.shape[0]
ind_X_train, ind_X_test, y_train, y_test = train_test_split(range(n_samples), y, stratify=y, test_size=0.3, random_state=RANDOM_SEED)

del all_data

list_model_description = list()
list_f1_measure_train = list()
list_precision_train = list()
list_recall_train = list()
list_accuracy_train = list()
list_f1_measure_test = list()
list_precision_test = list()
list_recall_test = list()
list_accuracy_test = list()

for ind_features in list_all_subsets(range(n_features)):
    n_features_in_subset = len(ind_features)
    if n_features_in_subset < 2:
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
    y_train = y[ind_X_train]
    y_test = y[ind_X_test]

    # train model
    model = LogisticRegression(penalty='l1', class_weight='balanced', solver='liblinear', random_state=RANDOM_SEED)
    model = model.fit(X_train, y_train)
    # TODO Store model?

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

    list_f1_measure_test.append(f1_score(y_test, predictions_test_binary))
    list_precision_test.append(precision_score(y_test, predictions_test_binary))
    list_recall_test.append(recall_score(y_test, predictions_test_binary))
    list_accuracy_test.append(accuracy_score(y_test, predictions_test_binary))

# write results to file
results = pd.DataFrame({
    'model_description': list_model_description,
    'f1_train': list_f1_measure_train,
    'precision_train': list_precision_train,
    'recall_train': list_recall_train,
    'accuracy_train': list_accuracy_train,
    'f1_test': list_f1_measure_test,
    'precision_test': list_precision_test,
    'recall_test': list_recall_test,
    'accuracy_test': list_accuracy_test
}).sort_values('f1_test', ascending=False)

results.to_csv(join(dir_output, 'predictors_screen.csv'), index=False)
