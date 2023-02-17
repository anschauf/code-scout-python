import os
import warnings
from os.path import join

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from src import ROOT_DIR
from test.sandbox_model_case_predictions.data_handler import load_data
from test.sandbox_model_case_predictions.utils import get_list_of_all_predictors, list_all_subsets
import xgboost

RANDOM_SEED = 42
COUNT_LIMIT = 10
TRAINING_SET_LIMIT = 100000 # -1 for no limit


dir_output = join(ROOT_DIR, 'results', 'xgboost_predictors_screen')
if not os.path.exists(dir_output):
    os.makedirs(dir_output)

revised_case_ids_filename = join(ROOT_DIR, 'resources', 'data', 'revised_case_ids.csv')
revised_cases_in_data = pd.read_csv(revised_case_ids_filename)

all_data = load_data(only_2_rows=True)
features_dir = join(ROOT_DIR, 'resources', 'features')
feature_filenames, encoders = get_list_of_all_predictors(all_data, features_dir, overwrite=False)
feature_names = sorted(list(feature_filenames.keys()))
n_features = len(feature_names)

y = revised_cases_in_data['is_revised'].values
n_samples = y.shape[0]
ind_X_train, ind_X_test, y_train, y_test = train_test_split(range(n_samples), y, stratify=y, test_size=0.3, random_state=RANDOM_SEED)

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

c_strengths = [1, 1000, 100, 10, 0.1, 0.01, 0.001]
# c_strengths = [1000, 100, 10, 1, 0.1, 0.01, 0.001]

for c in c_strengths:

    reached_limit = False
    count = 0

    for ind_features in list_all_subsets(range(n_features), reverse=True):
        if count >= COUNT_LIMIT:
            break
        else:
            count += 1
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
                # model = SVC(kernel='rbf', class_weight='balanced', C=c, random_state=RANDOM_SEED, probability=True)
                model = xgboost.XGBClassifier(
                    learning_rate=0.1,
                    max_depth=5,
                    n_estimators=5000,
                    subsample=0.5,
                    eval_metric='auc',
                    verbosity=1
                )

                if TRAINING_SET_LIMIT > 0:
                    X_train = X_train[:TRAINING_SET_LIMIT]
                    y_train = y_train[:TRAINING_SET_LIMIT]

                model = model.fit(X_train, y_train, verbose=True)

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

            logger.success(f'Training #{str(count)} done using c-value {str(c)}')
            results.to_csv(join(dir_output, f'svc_predictors_screen_c-{str(c)}.csv'))

logger.success('done')