import pickle
from os.path import join

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from sandbox_model_case_predictions.data_handler import load_data
from sandbox_model_case_predictions.experiments.random_forest import RANDOM_SEED
from sandbox_model_case_predictions.utils import get_list_of_all_predictors
from src import ROOT_DIR
from src.ml.explanation.tree.tree_explainer import TreeExplainer

# model_name = 'rf_5000'
model_name = 'rf_5000_depth5'


all_data = load_data(only_2_rows=True)
features_dir = join(ROOT_DIR, 'resources', 'features')
feature_filenames, encoders = get_list_of_all_predictors(all_data, features_dir, overwrite=False)
feature_names = sorted(list(feature_filenames.keys()))
n_features = len(feature_names)


all_feature_names = list()

for feature_idx in range(n_features):
    feature_name = feature_names[feature_idx]
    feature_name_wo_suffix = '_'.join(feature_name.split('_')[:-1])

    feature_filename = feature_filenames[feature_name]
    feature_values = np.load(feature_filename, mmap_mode='r', allow_pickle=False, fix_imports=False)

    if feature_name in encoders:
        encoder = encoders[feature_name]
        if isinstance(encoder, MultiLabelBinarizer):
            encoded_names = encoder.classes_

        else:
            if isinstance(encoder.categories, str) and encoder.categories == 'auto':
                encoded_names = encoder.categories_[0]
            else:
                encoded_names = encoder.categories

        all_feature_names.extend(f'{feature_name_wo_suffix}="{encoded_name}"' for encoded_name in encoded_names)

    else:
        all_feature_names.append(feature_name_wo_suffix)

logger.info(f"Loading the model '{model_name}' ...")
with open(join(ROOT_DIR, 'results', 'random_forest', f'{model_name}.pkl'), 'rb') as f:
    model = pickle.load(f)


# -----------------------------------------------------------------------------
# FEATURE IMPORTANCE
# -----------------------------------------------------------------------------
logger.info('Calculating the feature importance ...')

feature_importance = list(zip(all_feature_names, model.feature_importances_))

importances_df = (
    pd.DataFrame(feature_importance, columns=['feature', 'importance'])
    .sort_values(by=['importance', 'feature'], ascending=[False, True])
    .reset_index(drop=True)
)

print(importances_df[:20])

print(importances_df[~importances_df['feature'].str.startswith('month')][:20])


# -----------------------------------------------------------------------------
# FEATURE CONTRIBUTIONS
# -----------------------------------------------------------------------------
logger.info('Assembling test set ...')
revised_case_ids_filename = join(ROOT_DIR, 'resources', 'data', 'revised_case_ids.csv')
revised_cases_in_data = pd.read_csv(revised_case_ids_filename)

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
    feature_values = np.load(feature_filename, mmap_mode='r', allow_pickle=False, fix_imports=False)[ind_X_test, :]
    X.append(feature_values)

X_test = np.hstack(X)

logger.info('Calculating the feature contributions ...')
te = TreeExplainer(model, data=X_test, targets=y_test, n_jobs=1, verbose=True)
te.original_feature_names = all_feature_names
te.feature_names = all_feature_names


te.explain_feature_contributions(joint_contributions=False, ignore_non_informative_nodes=True)
te.plot_min_depth_distribution()

print('')

