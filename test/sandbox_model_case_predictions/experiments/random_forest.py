import os
import pickle
import warnings
from os.path import join

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import ROOT_DIR
from test.sandbox_model_case_predictions.data_handler import load_data
from test.sandbox_model_case_predictions.utils import get_list_of_all_predictors

RANDOM_SEED = 42


dir_output = join(ROOT_DIR, 'results', 'random_forest')
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

X = list()
features_in_subset = list()
for feature_idx in range(n_features):
    feature_name = feature_names[feature_idx]
    features_in_subset.append(feature_name)
    feature_filename = feature_filenames[feature_name]
    feature_values = np.load(feature_filename, mmap_mode='r', allow_pickle=False, fix_imports=False)
    X.append(feature_values)

X = np.hstack(X)

X_train = X[ind_X_train, :]
X_test = X[ind_X_test, :]

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

    normalizer = StandardScaler()
    X_train = normalizer.fit_transform(X_train)
    X_test = normalizer.transform(X_test)

    # train model
    model = RandomForestClassifier(
        n_estimators=5000,
        max_depth=None,  # TODO -> 5
        criterion='entropy',
        n_jobs=-1,
        random_state=RANDOM_SEED,
        class_weight='balanced'
    )

    model = model.fit(X_train, y_train)

    with open(join(ROOT_DIR, 'results', 'rf_5000.pkl'), 'wb') as f:
        pickle.dump(model, f, fix_imports=False)
