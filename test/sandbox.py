import pickle
import sys
from os.path import join

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MultiLabelBinarizer, OrdinalEncoder
from tqdm import tqdm

from sandbox_model_case_predictions.data_handler import load_data
from sandbox_model_case_predictions.experiments.random_forest import DISCARDED_FEATURES
from sandbox_model_case_predictions.utils import get_list_of_all_predictors, get_revised_case_ids
from src import ROOT_DIR
from src.ml.explanation.tree.utilities.parallel import compute_feature_contributions_from_tree

DISCARDED_FEATURES = list(DISCARDED_FEATURES) + ['vectorized_codes']


def load_model(model_indices: int | list[int] | None = None, *, return_ensemble: bool = False):
    logger.info(f"Loading the model ...")
    with open(join(ROOT_DIR, 'results', 'random_forest_only_reviewed',
                   # 'n_trees_1000-max_depth_None-min_samples_leaf_1_wVectors',
                   'n_trees_1000-max_depth_None-min_samples_leaf_1',
                   'rf_cv.pkl'), 'rb') as f:
        ensemble = pickle.load(f, fix_imports=False)

    if return_ensemble:
        return ensemble

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


def list_feature_names(*, discarded_features: list[str]):
    all_data = load_data(only_2_rows=True)
    features_dir = join(ROOT_DIR, 'resources', 'features')
    feature_filenames, encoders = get_list_of_all_predictors(all_data, features_dir, overwrite=False, log_ignored_features=False)
    feature_names = sorted(list(feature_filenames.keys()))

    feature_names = [feature_name for feature_name in feature_names
                     if not any(feature_name.startswith(discarded_feature) for discarded_feature in discarded_features)]

    all_feature_names = list()

    for feature_name in feature_names:
        feature_name_wo_suffix = '_'.join(feature_name.split('_')[:-1])

        if feature_name in encoders:
            encoder = encoders[feature_name]
            if isinstance(encoder, MultiLabelBinarizer):
                all_feature_names.extend(f'{feature_name_wo_suffix}="{encoded_name}"' for encoded_name in encoder.classes_)

            elif isinstance(encoder, OrdinalEncoder):
                options = '|'.join(encoder.categories_[0])
                all_feature_names.append(f'{feature_name_wo_suffix}_IN_[{options}]')

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


def get_features_and_targets(feature_names, feature_filenames, indices):
    logger.info('Assembling test set ...')
    revised_case_ids_filename = join(ROOT_DIR, 'resources', 'data', 'revised_case_ids.csv')
    # noinspection PyTypeChecker
    revised_cases_in_data = get_revised_case_ids(None, revised_case_ids_filename, overwrite=False)

    if indices is None:
        revised_cases_in_data = revised_cases_in_data[revised_cases_in_data['is_revised'] == 1]
    elif isinstance(indices, str) and indices == 'all':
        pass
    else:
        indices = np.where(np.in1d(revised_cases_in_data['index'], indices))[0]
        revised_cases_in_data = revised_cases_in_data.iloc[indices]

    y = revised_cases_in_data['is_revised'].values
    sample_indices = revised_cases_in_data['index'].values

    features = list()
    for feature_name in feature_names:
        feature_filename = feature_filenames[feature_name]
        feature_values = np.load(feature_filename, mmap_mode='r', allow_pickle=False, fix_imports=False)
        features.append(feature_values[sample_indices, :])
    features = np.hstack(features)

    logger.info(f'{features.shape=} {y.shape=}')

    return features, y, sample_indices


def calculate_feature_contributions(feature_names, feature_filenames):
    ensemble = load_model(return_ensemble=True)

    results_dir = join(ROOT_DIR, 'results', 'random_forest_only_reviewed', 'n_trees_1000-max_depth_None-min_samples_leaf_1')
    with open(join(results_dir, 'train_indices_per_model.pkl'), 'rb') as f:
        train_indices_per_model = pickle.load(f, fix_imports=False)

    all_features, all_y, all_sample_indices = get_features_and_targets(feature_names, feature_filenames, indices='all')

    predictions = np.zeros((all_features.shape[0], len(ensemble))) * np.nan
    for idx, model in enumerate(tqdm(ensemble)):
        all_predictions = model.predict_proba(all_features)[:, 1]

        train_indices = train_indices_per_model[idx]
        training_prediction_indices = np.in1d(all_sample_indices, train_indices, assume_unique=True)
        all_predictions[training_prediction_indices] = np.nan
        predictions[:, idx] = all_predictions

    del all_features
    del all_sample_indices

    prediction_probabilities = np.nanmean(predictions, axis=1)
    predictions = (prediction_probabilities >= 0.5).astype(int)
    # noinspection PyTypeChecker
    correct_predictions: np.ndarray = predictions == all_y

    logger.info(f'Number of correct predictions: {correct_predictions.sum()}/{correct_predictions.shape[0]}')

    tp_indices = np.where((predictions == 1) & (all_y == 1))[0]
    fp_indices = np.where((predictions == 1) & (all_y == 0))[0]
    fn_indices = np.where((predictions == 0) & (all_y == 1))[0]
    indices = np.unique(np.hstack((tp_indices, fp_indices, fn_indices)))

    features, y, sample_indices = get_features_and_targets(feature_names, feature_filenames, indices=indices)

    logger.info('Calculating the feature contributions ...')

    contributions_shape = (features.shape[0], features.shape[1], 2)
    contributions = np.zeros(contributions_shape, dtype=np.float64)
    contributions_n_evaluations = np.zeros(contributions_shape, dtype=np.int64)

    for model in ensemble:
        for estimator in tqdm(model.estimators_):
            tree_result = compute_feature_contributions_from_tree(
                estimator=estimator,
                data=features,
                contributions_shape=contributions_shape,
                features_split=None,
                joint_contributions=False,
                ignore_non_informative_nodes=True,
                contribution_measure='impurity'
            )

            contributions += tree_result['contributions']
            contributions_n_evaluations += tree_result['contributions_n_evaluations']
            del tree_result

    contributions[:, :, 1] *= -1

    average_contributions = contributions / contributions_n_evaluations
    average_contributions[~np.isfinite(average_contributions)] = 0.0


    # Average the contributions:
    # - across all the correct predictions for the positive label
    # - across all the incorrect predictions for the negative label
    tp_sample_indices = np.where(np.in1d(sample_indices, tp_indices))[0]
    fp_sample_indices = np.where(np.in1d(sample_indices, fp_indices))[0]
    fn_sample_indices = np.where(np.in1d(sample_indices, fn_indices))[0]

    n_tp_sample_indices = tp_sample_indices.shape[0]
    n_fp_sample_indices = fp_sample_indices.shape[0]
    n_fn_sample_indices = fn_sample_indices.shape[0]

    feature_contribution_tp = np.mean(average_contributions[tp_sample_indices, :, 1], axis=0)
    feature_contribution_fp = np.mean(average_contributions[fp_sample_indices, :, 1], axis=0)
    feature_contribution_fn = np.mean(average_contributions[fn_sample_indices, :, 0], axis=0)

    feature_values_tp = list()
    feature_values_fp = list()
    feature_values_fn = list()

    def _list_counts(values, unique_values, sample_indices, n):
        s = list()

        unique_values = sorted(unique_values, key=lambda x: -x)

        for v in unique_values:
            percentage = (values[sample_indices] == v).sum() / n * 100
            if v == 0:
                value = 'NO'
            elif v == 1:
                value = 'YES'
            else:
                raise ValueError(value)

            s.append(f"{value}={percentage:.1f}%")

        return ', '.join(s)

    for idx, feature_name in enumerate(tqdm(all_feature_names)):
        values = features[:, idx]
        is_categorical = '=' in feature_name

        if is_categorical:
            values = values.astype(int)
            all_values = np.unique(values)

            feature_values_tp.append(_list_counts(values, all_values, tp_sample_indices, n_tp_sample_indices))
            feature_values_fp.append(_list_counts(values, all_values, fp_sample_indices, n_fp_sample_indices))
            feature_values_fn.append(_list_counts(values, all_values, fn_sample_indices, n_fn_sample_indices))

        else:
            # feature_values_tp.append(f'{np.mean(values[tp_sample_indices]):.2f} [{np.min(values[tp_sample_indices]):.2f} - {np.max(values[tp_sample_indices]):.2f}]')
            # feature_values_fp.append(f'{np.mean(values[fp_sample_indices]):.2f} [{np.min(values[fp_sample_indices]):.2f} - {np.max(values[fp_sample_indices]):.2f}]')
            # feature_values_fn.append(f'{np.mean(values[fn_sample_indices]):.2f} [{np.min(values[fn_sample_indices]):.2f} - {np.max(values[fn_sample_indices]):.2f}]')
            feature_values_tp.append([f'{v:.2f}' for v in np.quantile(values[tp_sample_indices], (.25, .5, .75))])
            feature_values_fp.append([f'{v:.2f}' for v in np.quantile(values[fp_sample_indices], (.25, .5, .75))])
            feature_values_fn.append([f'{v:.2f}' for v in np.quantile(values[fn_sample_indices], (.25, .5, .75))])

    df = pd.DataFrame(np.vstack((
        feature_contribution_tp * 1000, feature_contribution_fp * 1000, feature_contribution_fn * 1000,
        feature_values_tp, feature_values_fp, feature_values_fn,
    )).T, index=all_feature_names, columns=[
        'contribution TP', 'contribution FP', 'contribution FN',
        f'avg value TP (n={n_tp_sample_indices})', f'avg value FP (n={n_fp_sample_indices})', f'avg value FN (n={n_fn_sample_indices})',
        ])

    df['contribution TP'] = df['contribution TP'].astype(np.float32)
    df['contribution FP'] = df['contribution FP'].astype(np.float32)
    df['contribution FN'] = df['contribution FN'].astype(np.float32)

    # df['delta'] = df['contribution TP'] - df['contribution FP'] + df['contribution FN']
    # df = df.sort_values(by='delta', ascending=False).reset_index()
    df = df.sort_values(by='contribution TP', ascending=False).reset_index()

    pd.set_option('display.precision', 5); print(df[:20])

    return df


if __name__ == '__main__':
    feature_names, all_feature_names, feature_filenames = list_feature_names(discarded_features=DISCARDED_FEATURES)

    model = load_model()
    feature_importance = calculate_feature_importance(model, all_feature_names)
    feature_importance.to_csv('feature_importance.csv', index=False)

    feature_contributions = calculate_feature_contributions(feature_names, feature_filenames)
    feature_contributions.to_csv('feature_contributions.csv', index=False)

    logger.success('done')
    sys.exit(0)
