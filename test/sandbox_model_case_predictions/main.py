from os import makedirs
from os.path import join, exists

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

from src import ROOT_DIR
from test.sandbox_model_case_predictions.data_handler import load_data
from test.sandbox_model_case_predictions.utils import get_list_of_all_predictors, all_subsets


RANDOM_SEED = 42
dir_output = join(ROOT_DIR, 'results', 'logistic_regression_predictors_screen')
if not exists(dir_output):
    makedirs(dir_output)

all_data = load_data()
list_X, list_X_labels, list_x_labels_predictor_title = get_list_of_all_predictors(all_data)

y = np.zeros((54645456, )) #TODO define correct labels from DB
n_samples = len(y)
ind_X_train, ind_X_test, y_train, y_test = train_test_split(range(n_samples), y, stratify=y, test_size=0.3, random_state=RANDOM_SEED)

list_model_description = list()
list_f1_measure_train = list()
list_precision_train = list()
list_recall_train = list()
list_accuracy_train = list()
list_f1_measure_test = list()
list_precision_test = list()
list_recall_test = list()
list_accuracy_test = list()
for ind_features in all_subsets(range(len(list_X))):
    if len(ind_features) > 0:
        # define data given selected predictors
        X = np.hstack([list_X[i-1] for i in ind_features])
        X_label = np.hstack([list_X_labels[i-1] for i in ind_features])
        list_model_description.append('|'.join(np.concatenate([list_X_labels[i-1] for i in ind_features])))
        logger.info(f'Training model {list_model_description[-1]}')

        X_train = X[ind_X_train]
        X_test = X[ind_X_test]

        # train model
        model = LogisticRegression(penalty='l1', class_weight='balanced', solver='liblinear', random_state=RANDOM_SEED)
        model = model.fit(X_train, y_train)

        # predict on train and test
        predictions_train = np.asarray([x[1] for x in model.predict_proba(X_train)])
        predictions_train_binary = [1 if (x > 0.5) else 0 for x in predictions_train]
        predictions_test = np.asarray([x[1] for x in model.predict_proba(X_test)])
        predictions_test_binary = [1 if (x > 0.5) else 0 for x in predictions_test]

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
        pd.DataFrame({
            'model_description': list_model_description,
            'f1_train': list_f1_measure_train,
            'precision_train': list_precision_train,
            'recall_train': list_recall_train,
            'accuracy_train': list_accuracy_train,
            'f1_test': list_f1_measure_test,
            'precision_test': list_precision_test,
            'recall_test': list_recall_test,
            'accuracy_test': list_accuracy_test
        }).to_csv(join(dir_output, 'predictors_screen.csv'), index=False)





print('')
