import os
import os
import shutil
import sys
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ndcg_score
from sklearn.model_selection import ParameterGrid, ShuffleSplit, StratifiedShuffleSplit
from tqdm import tqdm

from sandbox import list_feature_names
from src import ROOT_DIR
from src.files import load_revised_cases
from test.sandbox_model_case_predictions.data_handler import load_data
from test.sandbox_model_case_predictions.utils import create_performance_app_ground_truth, \
    create_predictions_output_performance_app, \
    get_list_of_all_predictors, get_revised_case_ids, \
    prepare_train_eval_test_split, RANDOM_SEED


def train_xgboost_only_reviewed_cases(*,
                                      do_recreate_ground_truth: bool = True,
                                      overwrite: bool = False
                                      ):

    revised_case_ids_filename = join(ROOT_DIR, 'resources', 'data', 'revised_case_ids.csv')
    all_data = load_data(only_2_rows=True)
    revised_cases_in_data = get_revised_case_ids(all_data, revised_case_ids_filename, overwrite=False)
    all_hospitals_and_years = revised_cases_in_data[revised_cases_in_data['is_revised'] == 1][['hospital', 'dischargeYear']].drop_duplicates().values.tolist()

    valid_hospitals_and_years = list()

    for LEAVE_ONE_OUT in all_hospitals_and_years:
        folder_name = f'global_performance_xgb/test_{LEAVE_ONE_OUT[0]}_{LEAVE_ONE_OUT[1]}'
        dir_ground_truth = join(ROOT_DIR, 'results', folder_name)

        if do_recreate_ground_truth:
            Path(dir_ground_truth).mkdir(parents=True, exist_ok=True)
            n_revised_cases = create_performance_app_ground_truth(dir_ground_truth, revised_cases_in_data, LEAVE_ONE_OUT[0], LEAVE_ONE_OUT[1])
            if n_revised_cases == 0:
                shutil.rmtree(dir_ground_truth, ignore_errors=True)
            else:
                valid_hospitals_and_years.append(LEAVE_ONE_OUT)

        else:
            if os.path.exists(dir_ground_truth):
                valid_hospitals_and_years.append(LEAVE_ONE_OUT)

    all_ground_truths = list()
    for dataset in valid_hospitals_and_years:
        filename = join(ROOT_DIR, 'results', 'global_performance_xgb', f'test_{dataset[0]}_{dataset[1]}', f'ground_truth_performance_app_case_ranking_{dataset[0]}_{dataset[1]}.csv')
        df = load_revised_cases(filename)

        df['delta_CW'] = df['CW_new'].astype(float) - df['CW_old'].astype(float)
        df = df[['combined_id', 'delta_CW']]
        all_ground_truths.append(df)

    all_ground_truths = pd.concat(all_ground_truths, ignore_index=True)
    revised_cases_in_data = pd.merge(revised_cases_in_data, all_ground_truths, left_on='id', right_on='combined_id', how='left')
    revised_cases_in_data['delta_CW'].fillna(0.0, inplace=True)

    discarded_features = {'hospital', 'month_admission', 'month_discharge', 'year_discharge', 'vectorized_codes'}

    _, all_feature_names, _ = list_feature_names(discarded_features=list(discarded_features))

    param_grid = {
        'n_estimators': [1000],
        'max_depth': [1],
        'eta': [0.01],
        'objective': [
            # 'rank:map',
            'rank:pairwise',
            # 'rank:ndcg',

            # 'binary:logistic',

            # 'reg:tweedie',
            # 'reg:squarederror',
            # 'reg:squaredlogerror',
            # 'reg:pseudohubererror',
            # 'reg:absoluteerror',
        ]
    }

    scoring = {
        'ndcg': 'NDCG'
    }

    features_dir = join(ROOT_DIR, 'resources', 'features')
    feature_filenames, encoders = get_list_of_all_predictors(all_data, features_dir, overwrite=False, log_ignored_features=False)
    feature_names = sorted(list(feature_filenames.keys()))
    feature_names = [feature_name for feature_name in feature_names if not any(feature_name.startswith(discarded_feature) for discarded_feature in discarded_features)]

    for params in ParameterGrid(param_grid):
        all_params = dict(params)
        all_params['n_jobs'] = -1
        all_params['random_state'] = RANDOM_SEED

        for LEAVE_ONE_OUT in valid_hospitals_and_years:
        # for LEAVE_ONE_OUT in [('KSW', 2020)]:
            folder_name = f'global_performance_xgb/test_{LEAVE_ONE_OUT[0]}_{LEAVE_ONE_OUT[1]}'
            filename = f'xgboost-n_estimators_{params["n_estimators"]}-max_depth_{params["max_depth"]}-eta_{str(params["eta"]).replace(".", "")}-objective_{params["objective"].replace(":", "_")}_stacked'

            results_dir = join(ROOT_DIR, 'results', folder_name, filename)
            if not os.path.exists(results_dir) or overwrite:
                shutil.rmtree(results_dir, ignore_errors=True)
                os.makedirs(results_dir, exist_ok=True)
            else:
                continue

            results_dir_test_predictions = join(ROOT_DIR, 'results', folder_name, 'TEST_PREDICTIONS')
            if not os.path.exists(results_dir_test_predictions):
                os.makedirs(results_dir_test_predictions, exist_ok=True)

            ind_train, ind_test, y_train, y_test, ind_hospital_leave_out, y_hospital_leave_out = \
                prepare_train_eval_test_split(revised_cases_in_data=revised_cases_in_data, hospital_leave_out=LEAVE_ONE_OUT[0], year_leave_out=LEAVE_ONE_OUT[1], dir_output=None,
                                              only_reviewed_cases=True)

            all_reviewed_case_indices = revised_cases_in_data[(revised_cases_in_data['is_revised'] == 1) | (revised_cases_in_data['is_reviewed'] == 1)]['index'].values
            ind_hospital_leave_out_reviewed_ind = np.in1d(ind_hospital_leave_out, all_reviewed_case_indices)
            ind_hospital_leave_out_reviewed = ind_hospital_leave_out[ind_hospital_leave_out_reviewed_ind]
            y_hospital_leave_out_reviewed = y_hospital_leave_out[ind_hospital_leave_out_reviewed_ind]

            ind_train_test = np.sort(np.hstack((ind_train, ind_test)))
            reviewed_cases = revised_cases_in_data[np.in1d(revised_cases_in_data['index'], ind_train_test)].reset_index(drop=True)

            if all_params['objective'].startswith('rank:'):
                groups = [tuple(group) for group in valid_hospitals_and_years]
                groups = {group: idx for idx, group in enumerate(groups)}
                reviewed_cases['group'] = reviewed_cases.apply(lambda row: groups.get((row['hospital'], row['dischargeYear']), -1), axis=1)
                reviewed_cases = reviewed_cases[reviewed_cases['group'] >= 0].reset_index(drop=True)

                groups_with_tp_and_tn = reviewed_cases.groupby('group')['is_revised'].agg(lambda x: len(set(x)))
                groups_with_tp_and_tn = groups_with_tp_and_tn[groups_with_tp_and_tn > 1].index.values
                n_groups = groups_with_tp_and_tn.shape[0]
                reviewed_cases = reviewed_cases[reviewed_cases['group'].isin(groups_with_tp_and_tn)].reset_index(drop=True)

                query_group_id = reviewed_cases['group'].values
                _, query_group_id = np.unique(query_group_id, return_inverse=True)
            else:
                query_group_id = np.zeros(reviewed_cases.shape[0], dtype=int)
                n_groups = 1

            sample_indices = reviewed_cases['index'].values


            if all_params['objective'].startswith('reg:'):
                y = reviewed_cases['delta_CW'].values
                y_hospital_leave_out = revised_cases_in_data.loc[ind_hospital_leave_out]['delta_CW'].values
            else:
                y = reviewed_cases['is_revised'].values

            # feature_values = np.load(feature_filename, mmap_mode='r', allow_pickle=False, fix_imports=False)
            # mdc_group_idx = list()
            # mdc_groups = list()
            # for mdc_idx in range(feature_values.shape[1]):
            #     idx = np.where(feature_values[reviewed_cases['index'].values, mdc_idx] == 1)[0]
            #     mdc_group_idx.append(idx)
            #     mdc_groups.append(np.tile([mdc_idx], idx.shape[0]))

            logger.info('Assembling training features ...')
            features = list()
            for feature_name in feature_names:
                feature_filename = feature_filenames[feature_name]
                feature_values = np.load(feature_filename, mmap_mode='r', allow_pickle=False, fix_imports=False)
                features.append(feature_values[sample_indices, :])
            features = np.hstack(features)

            logger.info('Assembling test features ...')
            test_features_all = list()
            test_features_reviewed = list()
            for feature_name in feature_names:
                feature_filename = feature_filenames[feature_name]
                feature_values = np.load(feature_filename, mmap_mode='r', allow_pickle=False, fix_imports=False)
                test_features_all.append(feature_values[ind_hospital_leave_out, :])
                test_features_reviewed.append(feature_values[ind_hospital_leave_out_reviewed, :])
            test_features_all = np.hstack(test_features_all)
            test_features_reviewed = np.hstack(test_features_reviewed)

            if all_params['objective'].startswith('reg:'):
                cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=RANDOM_SEED)
            else:
                cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=RANDOM_SEED)
            n_models = cv.n_splits
            logger.info(f'Training an xgboost with {params=} ...')

            scores = dict()
            scores['estimator'] = list()
            scores['test_predictions'] = np.zeros((test_features_all.shape[0], n_models), dtype=np.float32)
            scores['train_ndcg'] = np.zeros((n_groups, n_models), dtype=np.float32)
            scores['eval_ndcg'] = np.zeros_like(scores['train_ndcg'])
            scores['test_ndcg'] = np.zeros(n_models, dtype=np.float32)
            scores['test_ndcg_reviewed'] = np.zeros_like(scores['test_ndcg'])

            for split_idx, (ind_train, ind_eval) in enumerate(tqdm(cv.split(features, y), total=n_models)):
                ind_train = np.sort(ind_train)
                ind_eval = np.sort(ind_eval)
                train_query_group_id = query_group_id[ind_train]
                train_labels = y[ind_train]

                if all_params['objective'].startswith('rank:'):
                    df = pd.DataFrame({
                        'train_query_group_id': train_query_group_id,
                        'train_labels': train_labels,
                        'ind_train': ind_train,
                    }).sort_values(['train_query_group_id', 'train_labels'])

                    groups_with_tp_and_tn = df.groupby('train_query_group_id')['train_labels'].agg(lambda x: len(set(x)))
                    groups_with_tp_and_tn = groups_with_tp_and_tn[groups_with_tp_and_tn > 1].index.values
                    n_groups = groups_with_tp_and_tn.shape[0]
                    df = df[df['train_query_group_id'].isin(groups_with_tp_and_tn)].reset_index(drop=True)

                    train_query_group_id = df['train_query_group_id'].values
                    _, train_query_group_id = np.unique(train_query_group_id, return_inverse=True)
                    train_labels = df['train_labels'].values
                    ind_train = df['ind_train'].values

                    rf_model = RandomForestClassifier(
                        n_estimators=1000,
                        max_depth=10,
                        min_samples_leaf=400,
                        min_samples_split=2,
                        class_weight='balanced',
                        oob_score=False,
                        min_impurity_decrease=np.finfo(np.float32).eps,  # the smallest positive number, so that it is not 0
                        criterion='entropy', n_jobs=-1, random_state=RANDOM_SEED,
                    )
                    rf_model.fit(features[ind_train, :], train_labels)
                    leaves = rf_model.apply(features[ind_train, :])

                    model = xgb.XGBRanker(**all_params)
                    model.fit(leaves, train_labels, qid=train_query_group_id)
                    # model.fit(features[ind_train, :], train_labels, qid=train_query_group_id)

                else:
                    labels, sample_weights = np.unique(y[ind_train], return_counts=True)
                    sample_weights = sample_weights.astype(float) / sample_weights.sum()
                    balanced_sample_weights = {l: 1 - sw for l, sw in zip(labels, sample_weights)}
                    sample_weights = np.array([balanced_sample_weights[l] for l in y[ind_train]], dtype=float)

                    if all_params['objective'].startswith('reg:'):
                        model = xgb.XGBRegressor(**all_params)
                        model.fit(features[ind_train, :], y[ind_train], sample_weight=sample_weights)

                    elif all_params['objective'].startswith('binary:'):
                        model = xgb.XGBClassifier(**all_params)
                        model.fit(features[ind_train, :], y[ind_train], sample_weight=sample_weights)

                    else:
                        raise ValueError(f'Unknown model for objective "{all_params["objective"]}"')

                scores['estimator'].append(model)

                def predict(features_mat):
                    return model.predict(rf_model.apply(features_mat))

                train_predictions = predict(features[ind_train, :])
                train_scores = score_ranking(train_predictions, train_labels, train_query_group_id, n_groups)
                scores['train_ndcg'][:, split_idx] = train_scores

                eval_predictions = predict(features[ind_eval, :])
                eval_scores = score_ranking(eval_predictions, y[ind_eval], query_group_id[ind_eval], n_groups)
                scores['eval_ndcg'][:, split_idx] = eval_scores

                test_predictions_all = predict(test_features_all)
                test_scores_all = score_ranking(test_predictions_all, y_hospital_leave_out, np.zeros_like(y_hospital_leave_out, dtype=int), 1)
                scores['test_ndcg'][split_idx] = test_scores_all[0]
                scores['test_predictions'][:, split_idx] = test_predictions_all  # Will be stored for the "case ranking" app

                test_predictions_reviewed = predict(test_features_reviewed)
                test_scores_reviewed = score_ranking(test_predictions_reviewed, y_hospital_leave_out_reviewed, np.zeros_like(y_hospital_leave_out_reviewed, dtype=int), 1)
                scores['test_ndcg_reviewed'][split_idx] = test_scores_reviewed[0]

                # TODO Study the features
                # feature_contributions = model._Booster.predict(eval_data, pred_contribs=True)
                # feature_interactions = model._Booster.predict(eval_data, pred_interactions=True)
                # pd.DataFrame(list(zip(model.feature_importances_, feature_ids))).sort_values(0, ascending=False)[:20]

            logger.info('--- Average performance ---')
            performance_log = list()
            performance_log.append(f'# revised: {int(y.sum())}')
            performance_log.append(f'# cases: {y.shape[0]}')

            longest_scorer_name = max(len(name) for name in scoring.values())
            for metric_name, descriptive_metric_name in scoring.items():
                train_metric = scores[f'train_{metric_name}']
                eval_metric = scores[f'eval_{metric_name}']
                test_metric_all = scores[f'test_{metric_name}']
                test_metric_reviewed = scores[f'test_{metric_name}_reviewed']
                pad = ' ' * (longest_scorer_name - len(descriptive_metric_name) + 1)

                msg = f'{descriptive_metric_name}:{pad}train {np.nanmean(train_metric):.6f}, eval {np.nanmean(eval_metric):.6f}, test (all) {np.nanmean(test_metric_all):.6f}, test (reviewed) {np.nanmean(test_metric_reviewed):.6f}'
                performance_log.append(msg)
                logger.info(msg)

            with open(join(results_dir, 'performance.txt'), 'w') as f:
                f.writelines('\n'.join(performance_log))

            # logger.info('Storing models ...')
            # with open(join(results_dir, 'rf_cv.pkl'), 'wb') as f:
            #     pickle.dump(scores['estimator'], f, fix_imports=False)

            # f_importances = list()
            # for estimator in scores['estimator']:
            #     f_importances.append(estimator.feature_importances_)
            # f_importances = np.vstack(f_importances)
            # mean_feature = np.nanmean(f_importances, axis=0)
            # std_feature = np.nanstd(f_importances, axis=0)
            #
            # pd.DataFrame({
            #     'feature': all_feature_names,
            #     'feature_importance_mean': mean_feature,
            #     'feature_importance_std': std_feature
            #     }) \
            #     .sort_values(by='feature_importance_mean', ascending=False) \
            #     .to_csv(join(results_dir, 'feature_importances_random_forest.csv'), index=False)

            logger.info(f"Storing predictions for '{LEAVE_ONE_OUT[0]}' in {LEAVE_ONE_OUT[1]} ...")
            filename_output = join(results_dir_test_predictions, f'{filename}-{LEAVE_ONE_OUT[0]}-{LEAVE_ONE_OUT[1]}.csv')
            hospital_data = revised_cases_in_data[(revised_cases_in_data['hospital'] == LEAVE_ONE_OUT[0]) & (revised_cases_in_data['dischargeYear'] == LEAVE_ONE_OUT[1])]
            case_ids = hospital_data['id'].values

            predictions = np.nanmean(scores['test_predictions'], axis=1)
            predictions = predictions - np.nanmin(predictions)
            predictions = predictions / np.nanmax(predictions)

            create_predictions_output_performance_app(filename_output, case_ids, predictions)

    logger.success('done')


def score_ranking(predictions, labels, groups, n_groups):
    ndcg_score_vec = np.zeros(n_groups, dtype=np.float32) * np.nan

    for group in np.unique(groups):
        group_idx = groups == group
        group_predictions = predictions[group_idx]
        group_labels = labels[group_idx]

        # Normalized Discounted Cumulative Gain
        try:
            ndcg_score_1 = ndcg_score(y_true=[group_labels], y_score=[group_predictions])
            ndcg_score_vec[group] = ndcg_score_1
        except ValueError:
            pass

    return ndcg_score_vec


if __name__ == '__main__':
    train_xgboost_only_reviewed_cases(
        do_recreate_ground_truth=False,
        overwrite=False,
    )

    sys.exit(0)
