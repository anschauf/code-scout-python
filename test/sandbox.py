import gc
import pickle
from os.path import join

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from sandbox_model_case_predictions.data_handler import load_data
from sandbox_model_case_predictions.experiments.random_forest import DISCARDED_FEATURES
from sandbox_model_case_predictions.utils import get_list_of_all_predictors, get_revised_case_ids
from src import ROOT_DIR
from src.ml.explanation.tree.utilities.numerical import divide0
from src.ml.explanation.tree.utilities.parallel import compute_feature_contributions_from_tree


def load_model(model_indices: int | list[int] | None = None):
    logger.info(f"Loading the model ...")
    with open(join(ROOT_DIR, 'results', 'random_forest_only_reviewed',
                   'n_trees_5000-max_depth_None-min_samples_leaf_1',
                   'rf_cv.pkl'), 'rb') as f:
        ensemble = pickle.load(f, fix_imports=False)

    if model_indices is None:
        model_indices = list(range(len(ensemble)))
    elif isinstance(model_indices, int):
        model_indices = [model_indices]

    # Combine the models
    model = ensemble[model_indices[0]]
    for idx in model_indices[1:]:
        other_model = ensemble[model_indices[idx]]
        model.estimators_ += other_model.estimators_
    model.n_estimators = len(model.estimators_)

    return model


def list_feature_names():
    all_data = load_data(only_2_rows=True)
    features_dir = join(ROOT_DIR, 'resources', 'features')
    feature_filenames, encoders = get_list_of_all_predictors(all_data, features_dir, overwrite=False)
    feature_names = sorted(list(feature_filenames.keys()))

    feature_names = [feature_name for feature_name in feature_names
                     if not any(feature_name.startswith(discarded_feature) for discarded_feature in DISCARDED_FEATURES)]

    all_feature_names = list()

    for feature_name in feature_names:
        feature_name_wo_suffix = '_'.join(feature_name.split('_')[:-1])

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

    return feature_names, all_feature_names, feature_filenames


def calculate_feature_importance(model, all_feature_names):
    logger.info('Calculating the feature importance ...')

    feature_importance = list(zip(all_feature_names, model.feature_importances_))

    importances_df = (
        pd.DataFrame(feature_importance, columns=['feature', 'importance'])
        .sort_values(by=['importance', 'feature'], ascending=[False, True])
        .reset_index(drop=True)
    )

    print(importances_df[:20])

    return importances_df


def get_features_and_targets(feature_names, feature_filenames):
    logger.info('Assembling test set ...')
    revised_case_ids_filename = join(ROOT_DIR, 'resources', 'data', 'revised_case_ids.csv')
    # noinspection PyTypeChecker
    revised_cases_in_data = get_revised_case_ids(None, revised_case_ids_filename, overwrite=False)
    revised_cases_in_data = revised_cases_in_data[revised_cases_in_data['is_revised'] == 1]
    y = revised_cases_in_data['is_revised'].values
    sample_indices = revised_cases_in_data['index'].values

    features = list()
    for feature_name in feature_names:
        feature_filename = feature_filenames[feature_name]
        feature_values = np.load(feature_filename, mmap_mode='r', allow_pickle=False, fix_imports=False)
        features.append(feature_values[sample_indices, :])
    features = np.hstack(features)

    logger.info(f'{features.shape=} {y.shape=}')

    return features, y


def calculate_feature_contributions(model, features, y):
    gc.collect()
    logger.info('Calculating the feature contributions ...')

    n = model.n_estimators
    contributions_shape = (features.shape[0], features.shape[1], 2)

    contributions = np.zeros((n, ), dtype=object)
    contributions_n_evaluations = np.zeros((n, ), dtype=object)

    for i_tree, estimator in tqdm(enumerate(model.estimators_), total=n):
        tree_result = compute_feature_contributions_from_tree(
            estimator=estimator,
            data=features,
            contributions_shape=contributions_shape,
            features_split=None,
            joint_contributions=False,
            ignore_non_informative_nodes=True
        )

        contributions[i_tree] = tree_result['contributions']
        contributions_n_evaluations[i_tree] = tree_result['contributions_n_evaluations']
        del tree_result

    contributions = divide0(np.sum(np.stack(contributions, axis=3), axis=3),
                            np.sum(np.stack(contributions_n_evaluations, axis=3), axis=3),
                            replace_with=np.nan)

    prediction_probabilities = model.predict_proba(features)
    predictions = np.argmax(prediction_probabilities, axis=1)
    correct_predictions = predictions == y

    logger.info(f'Number of correct predictions: {correct_predictions.sum()}/{correct_predictions.shape[0]}')

    # Average the contributions:
    # - across all the correct predictions for the positive label
    # - across all the incorrect predictions for the negative label
    feature_contribution_correct = np.mean(contributions[correct_predictions, :, 1], axis=0)
    feature_contribution_incorrect = np.mean(contributions[~correct_predictions, :, 0], axis=0)

    correct_col = 'contribution to correct prediction'
    incorrect_col = 'contribution to wrong prediction'
    delta_col = 'delta'

    df = pd.DataFrame(np.vstack((feature_contribution_correct, feature_contribution_incorrect)).T, index=all_feature_names, columns=[correct_col, incorrect_col])
    # df = df[df[correct_col] > 0]
    df[delta_col] = df[correct_col] - df[incorrect_col]
    df *= 1000
    df = df.sort_values(by=delta_col, ascending=False).reset_index()
    pd.set_option('display.precision', 5); print(df[:20])

    # df2 = pd.DataFrame(te.contributions[0, :, 1].reshape(-1, 1), index=te.feature_names, columns=['revised'])
    # df2['feature_value'] = X_test[0, :]
    # df2.sort_values(by='revised', ascending=False, inplace=True)
    # print(df2[:20])

    return df


if __name__ == '__main__':
    feature_names, all_feature_names, feature_filenames = list_feature_names()

    model = load_model()

    # feature_importance = calculate_feature_importance(model, all_feature_names)
    # feature_importance.to_csv('feature_importance.csv', index=False)

    features, y = get_features_and_targets(feature_names, feature_filenames)
    feature_contributions = calculate_feature_contributions(model, features, y)
    feature_contributions.to_csv('feature_contributions.csv', index=False)
