import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from beartype import beartype
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, ndcg_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedShuffleSplit
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm

from sandbox_model_case_predictions.experiments.xgboost_leave_one_out_cv_hyperparameter_screen import score_ranking
from sandbox_model_case_predictions.utils import create_predictions_output_performance_app, RANDOM_SEED
from src import ROOT_DIR
from src.apps.case_ranking import create_rankings_of_revised_cases
from src.service.bfs_cases_db_service import get_clinics, get_sociodemographics_by_case_id
from src.service.database import Database

tqdm.pandas()


# noinspection PyShadowingNames,PyUnboundLocalVariable,PyBroadException
def train_random_forest(*,
                        model_to_train: str,
                        overwrite: bool = False
                        ):

    # revised_case_ids_filename = os.path.join(ROOT_DIR, 'resources', 'data', 'revised_case_ids.csv')
    # revised_cases_in_data = get_revised_case_ids(None, revised_case_ids_filename, overwrite=False)
    # features_dir = os.path.join(ROOT_DIR, 'resources', 'features')
    # training_feature_filename = os.path.join(features_dir, f'mind_bend_ksw_2020.csv')
    # training_features_df = pd.read_csv(training_feature_filename)
    # training_features_df['case_id'] = training_features_df['case_id'].astype(str)
    # mind_bend_case_ids = set(training_features_df['case_id'].values)
    # ksw_2020 = revised_cases_in_data[(~revised_cases_in_data['id'].isin(mind_bend_case_ids)) & (revised_cases_in_data['hospital'] == 'KSW') & (revised_cases_in_data['dischargeYear'] == 2020)]

    with Database() as db:
        clinics_df = get_clinics(db.session)

    if model_to_train not in ('RF', 'xgboost', 'ranker'):
        raise ValueError('Only "RF", "xgboost" and "ranker" are allowed')

    discarded_features = ['sum_p_suggestions_per_case', 'sum_support_suggestions_per_case']
    label_column = 'n_added_codes_in_suggestions'

    if model_to_train == 'RF':
        param_grid = {
            'n_estimators': [1000],
            'max_depth': [1, 2, 5, 10],
        }
    elif model_to_train in ('xgboost', 'ranker'):
        param_grid = {
            # 'n_estimators': [1000, 2000, 3000],
            'n_estimators': [1000],
            # 'max_depth': [3, 4, 5, 6, 7, 8, 9],
            'max_depth': [3, 4, 5, 6],
            'eta': [0.005, 0.0075, 0.01],
            # 'subsample': [0.3, 0.5],  # "Subsample ratio of the training instance"
            # 'colsample_bytree': [0.5, 0.75],  # "Subsample ratio of columns when constructing each tree"
            # 'colsample_bylevel': [0.5, 0.75],  # "Subsample ratio of columns when constructing each level"
            # 'colsample_bynode': [0.5, 0.75],  # "Subsample ratio of columns when constructing each split"
        }

        if model_to_train == 'xgboost':
            param_grid['objective'] = ['binary:logistic']
        else:
            # param_grid['objective'] = ['rank:map', 'rank:pairwise', 'rank:ndcg']
            param_grid['objective'] = ['rank:pairwise']
            model_to_train = 'xgboost'

    scoring = {
        'NDCG': 'ndcg',
        'AUROC': 'roc_auc',
        'AUPRC': 'average_precision',
        'precision': 'precision',
        'recall': 'recall',
        'F1': 'f1',
    }

    features_dir = os.path.join(ROOT_DIR, 'resources', 'features')
    training_feature_filename = os.path.join(features_dir, f'mind_bend_reviewed.csv')

    for LEAVE_ONE_OUT in [
        ('KSW', 2020),
        # ('HLZ', 2020),
    ]:
        test_feature_filename = os.path.join(features_dir, f'mind_bend_{LEAVE_ONE_OUT[0].lower()}_{LEAVE_ONE_OUT[1]}.csv')
        test_features_df = pd.read_csv(test_feature_filename) \
            .sort_values(['is_case_revised', 'case_id', 'target_drg_cost_weight'], ascending=[False, True, True]) \
            .reset_index(drop=False)

        all_test_case_ids = set(test_features_df['case_id'].values)
        training_features_df = pd.read_csv(training_feature_filename)
        training_features_df = training_features_df[~training_features_df['case_id'].isin(all_test_case_ids)] \
            .sort_values(['is_case_revised', 'case_id', 'target_drg_cost_weight']).reset_index(drop=True) \
            .reset_index(drop=False) \
            .sort_values('index').reset_index(drop=True)

        logger.info('Reading from DB ...')
        with Database() as db:
            training_case_ids = list({str(case_id) for case_id in training_features_df['case_id'].values.tolist()})
            training_sociodemographics = get_sociodemographics_by_case_id(training_case_ids, db.session)
            logger.info(f'Read {training_sociodemographics.shape[0]} training cases ...')

        logger.info('Filtering clinics ...')
        training_sociodemographics = pd.merge(training_sociodemographics[['case_id', 'clinic_id']], clinics_df[['clinic_id', 'clinic_code']], on='clinic_id', how='inner').drop(columns='clinic_id')
        training_sociodemographics = training_sociodemographics[training_sociodemographics['clinic_code'].isin(['M100', 'M200'])]
        training_features_df = pd.merge(training_features_df, training_sociodemographics['case_id'].astype(int), on='case_id', how='inner') \
            .drop(columns='index').reset_index(drop=True).reset_index(drop=False)

        # Map each case ID to the list of rows where
        all_case_id_to_index_dict = (training_features_df
                                     .groupby('case_id')
                                     ['index'].agg(list)
                                     .to_dict())

        training_label_per_case = training_features_df.groupby('case_id')[['is_case_revised', 'index']].agg({
            'index': lambda x: sorted(list(x))[0],
            'is_case_revised': lambda x: list(x)[0]
        }).sort_values('index').reset_index(drop=False)
        all_training_case_ids = training_label_per_case['case_id'].values
        all_training_labels = training_label_per_case['is_case_revised'].values.astype(int)

        features_and_labels_df = training_features_df.drop(
            columns=['case_id', 'index', 'is_case_revised', 'target_drg', 'does_contain_revised_drg', 'n_added_codes',
                     'n_revised_codes_in_suggestions'] + discarded_features)
        all_features = features_and_labels_df.drop(columns=label_column).values
        feature_names = list(features_and_labels_df.drop(columns=label_column).columns)
        all_labels = (features_and_labels_df[label_column] > 0).values.astype(int)

        test_features_and_labels_df = test_features_df.drop(columns=['case_id', 'index', 'is_case_revised', 'target_drg', 'does_contain_revised_drg', 'n_added_codes', 'n_revised_codes_in_suggestions'] + discarded_features)
        test_features = test_features_and_labels_df.drop(columns=label_column).values
        test_y = (test_features_and_labels_df[label_column] > 0).values.astype(int)

        test_predictions_df = test_features_df[['index', 'case_id', 'is_case_revised', label_column]].copy()
        test_predictions_df['is_correct_suggestion'] = test_predictions_df[label_column] > 0

        all_performance_info_for_left_out = list()
        folder_name = f'mind_bend_reviewed/test_{LEAVE_ONE_OUT[0]}_{LEAVE_ONE_OUT[1]}'

        for params in ParameterGrid(param_grid):
            all_params = dict(params)
            all_params['n_jobs'] = -1
            all_params['random_state'] = RANDOM_SEED

            is_ranking_model = model_to_train == 'xgboost' and all_params['objective'].startswith('rank:')

            if is_ranking_model:
                training_features_df_per_params = training_features_df[training_features_df['is_case_revised']] \
                    .drop(columns='index').reset_index(drop=True) \
                    .reset_index(drop=False)

                training_label_per_case_per_params = training_features_df_per_params.groupby('case_id')[['is_case_revised', 'index']].agg({
                    'index': lambda x: sorted(list(x))[0],
                    'is_case_revised': lambda x: list(x)[0]
                }).sort_values('index').reset_index(drop=False)
                all_case_ids = training_label_per_case_per_params['case_id'].values
                training_labels = training_label_per_case_per_params['is_case_revised'].values.astype(int)

                features_and_labels_df = training_features_df_per_params.drop(
                    columns=['case_id', 'index', 'is_case_revised', 'target_drg', 'does_contain_revised_drg', 'n_added_codes',
                             'n_revised_codes_in_suggestions'] + discarded_features)
                features = features_and_labels_df.drop(columns=label_column).values
                y = (features_and_labels_df[label_column] > 0).values.astype(int)

                case_id_to_index_dict = (training_features_df_per_params
                                         .groupby('case_id')
                                         ['index'].agg(list)
                                         .to_dict())

                encoder = OrdinalEncoder(categories=[all_case_ids.astype(str)])
                case_id_groups = encoder.fit_transform(training_features_df_per_params['case_id'].values.astype(str).reshape(-1, 1))[:, 0].astype(int)
                n_ranking_groups = np.unique(case_id_groups).shape[0]

                del training_features_df_per_params, training_label_per_case_per_params

            else:
                all_case_ids = all_training_case_ids.copy()
                training_labels = all_training_labels.copy()
                features = all_features.copy()
                y = all_labels.copy()
                case_id_to_index_dict = dict(all_case_id_to_index_dict)
                case_id_groups = np.zeros_like(all_case_ids, dtype=int)
                n_ranking_groups = None

            dummy_features_arr = np.zeros((training_labels.shape[0], 1), dtype=np.float32)

            if model_to_train == 'RF':
                all_params['min_impurity_decrease'] = np.finfo(np.float32).eps  # the smallest positive number, so that it is not 0
                all_params['class_weight'] = 'balanced'
                all_params['oob_score'] = False
                all_params['criterion'] = 'entropy'

            filename = [model_to_train.lower()]
            for param_name, param_value in params.items():
                param_value = str(param_value).replace(":", "_").replace(".", "")
                filename.append(f'{param_name}_{param_value}')
            filename = '-'.join(filename)

            results_dir = os.path.join(ROOT_DIR, 'results', folder_name, filename)
            if not os.path.exists(results_dir) or overwrite:
                shutil.rmtree(results_dir, ignore_errors=True)
            else:
                continue

            performance_info_for_left_out = dict()
            performance_info_for_left_out['model'] = model_to_train
            performance_info_for_left_out['params'] = all_params

            cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=RANDOM_SEED)
            n_models = cv.n_splits
            performance_info_for_left_out['n_cv_models'] = n_models
            logger.info(f'Training "{model_to_train}" with {params=} ...')

            performance_info_for_left_out['performance'] = dict()
            for metric in scoring.keys():
                performance_info_for_left_out['performance'][f'train_{metric}'] = list(np.zeros(n_models, dtype=np.float32) * np.nan)
                performance_info_for_left_out['performance'][f'eval_{metric}'] = list(np.zeros(n_models, dtype=np.float32) * np.nan)
                performance_info_for_left_out['performance'][f'test_{metric}'] = list(np.zeros(n_models, dtype=np.float32) * np.nan)

            cv_models = list()
            all_test_predictions = np.zeros((test_features.shape[0], n_models), dtype=np.float32)

            for split_idx, (case_id_ind_train, case_id_ind_eval) in enumerate(tqdm(cv.split(dummy_features_arr, training_labels), total=n_models)):
                case_id_ind_train = np.sort(case_id_ind_train)
                case_id_ind_eval = np.sort(case_id_ind_eval)

                ind_train = np.unique(np.hstack([case_id_to_index_dict[all_case_ids[i]] for i in case_id_ind_train]))
                ind_eval = np.unique(np.hstack([case_id_to_index_dict[all_case_ids[i]] for i in case_id_ind_eval]))

                train_features = features[ind_train, :]
                eval_features = features[ind_eval, :]
                train_labels = y[ind_train]
                eval_labels = y[ind_eval]

                train_predictions_df = training_features_df.loc[ind_train][['index', 'case_id', 'is_case_revised', label_column]].copy()
                train_predictions_df['is_correct_suggestion'] = train_predictions_df[label_column] > 0
                eval_predictions_df = training_features_df.loc[ind_eval][['index', 'case_id', 'is_case_revised', label_column]].copy()
                eval_predictions_df['is_correct_suggestion'] = eval_predictions_df[label_column] > 0

                train_query_id = None
                eval_query_id = None
                model = None

                if model_to_train == 'RF':
                    model = RandomForestClassifier(**all_params)
                    model.fit(train_features, train_labels)

                elif model_to_train == 'xgboost':
                    if is_ranking_model:
                        # The query_id corresponds to individual cases
                        _, train_query_id = np.unique(case_id_groups[ind_train], return_inverse=True)
                        _, eval_query_id = np.unique(case_id_groups[ind_eval], return_inverse=True)

                        model = xgb.XGBRanker(**all_params)
                        # From the documentation @ https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.DMatrix:
                        # "For ranking task, weights are per-group. In ranking task, one weight is assigned to each
                        # group (not each data point). This is because we only care about the relative ordering of data
                        # points within each group, so it doesn’t make sense to assign weights to individual data points."
                        model.fit(train_features, train_labels, qid=train_query_id)

                    else:
                        _, class_counts = np.unique(train_labels, return_counts=True)
                        class_counts = class_counts / np.sum(class_counts)
                        class_weights = 1 - class_counts

                        sample_weights = np.zeros_like(train_labels, dtype=np.float32)
                        sample_weights[train_labels == 0] = class_weights[0]
                        sample_weights[train_labels == 1] = class_weights[1]

                        model = xgb.XGBClassifier(**all_params)
                        model.fit(train_features, train_labels, sample_weight=sample_weights)

                cv_models.append(model)

                if is_ranking_model:
                    train_score = model.predict(train_features)
                    eval_score = model.predict(eval_features)
                    test_score = model.predict(test_features)
                else:
                    train_score = model.predict_proba(train_features)[:, 1]
                    eval_score = model.predict_proba(eval_features)[:, 1]
                    test_score = model.predict_proba(test_features)[:, 1]

                train_scores_df = train_predictions_df.copy()
                train_scores_df['score'] = train_score
                train_classification_scores_df = train_scores_df.groupby('case_id').agg({
                    'index': lambda x: sorted(list(x))[0],
                    'is_case_revised': lambda x: list(x)[0],
                    'score': lambda x: np.max(x)
                }).sort_values('index')
                train_y_true = train_classification_scores_df['is_case_revised'].values.astype(int)

                eval_scores_df = eval_predictions_df.copy()
                eval_scores_df['score'] = eval_score
                eval_classification_scores_df = eval_scores_df.groupby('case_id').agg({
                    'index': lambda x: sorted(list(x))[0],
                    'is_case_revised': lambda x: list(x)[0],
                    'score': lambda x: np.max(x)
                }).sort_values('index')
                eval_y_true = eval_classification_scores_df['is_case_revised'].values.astype(int)

                all_test_predictions[:, split_idx] = test_score
                test_scores_df = test_predictions_df.copy()
                test_scores_df['score'] = test_score
                test_classification_scores_df = test_scores_df.groupby('case_id').agg({
                    'index': lambda x: sorted(list(x))[0],
                    'is_case_revised': lambda x: list(x)[0],
                    'score': lambda x: np.max(x)
                }).sort_values('index')
                test_y_true = (test_classification_scores_df['is_case_revised']).values.astype(int)

                if is_ranking_model:
                    train_y_pred = (train_classification_scores_df['score'] > 0).values.astype(int)
                    eval_y_pred = (eval_classification_scores_df['score'] > 0).values.astype(int)
                    test_y_pred = (test_classification_scores_df['score'] > 0).values.astype(int)
                else:
                    train_y_pred = (train_classification_scores_df['score'] > 0.5).values.astype(int)
                    eval_y_pred = (eval_classification_scores_df['score'] > 0.5).values.astype(int)
                    test_y_pred = (test_classification_scores_df['score'] > 0.5).values.astype(int)

                for metric_name, metric_str in scoring.items():
                    if metric_str == 'ndcg':
                        if is_ranking_model:
                            train_metric = np.nanmean(score_ranking(train_score, train_labels, train_query_id, n_ranking_groups))
                            eval_metric = np.nanmean(score_ranking(eval_score, eval_labels, eval_query_id, n_ranking_groups))

                        else:
                            train_metric = ndcg_score(y_true=[train_labels], y_score=[train_score])
                            eval_metric = ndcg_score(y_true=[eval_labels], y_score=[eval_score])

                        test_metric = ndcg_score(y_true=[test_y], y_score=[test_score])

                    else:
                        if metric_str == 'roc_auc':
                            scoring_function = roc_auc_score
                        elif metric_str == 'average_precision':
                            scoring_function = average_precision_score
                        elif metric_str == 'precision':
                            scoring_function = precision_score
                        elif metric_str == 'recall':
                            scoring_function = recall_score
                        elif metric_str == 'f1':
                            scoring_function = f1_score
                        else:
                            raise ValueError(f'Unknown scoring metric {metric_str}')

                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            try:
                                train_metric = scoring_function(train_y_true, train_y_pred)
                            except ValueError:
                                train_metric = np.nan

                            try:
                                eval_metric = scoring_function(eval_y_true, eval_y_pred)
                            except ValueError:
                                eval_metric = np.nan

                            try:
                                test_metric = scoring_function(test_y_true, test_y_pred)
                            except ValueError:
                                test_metric = np.nan

                    performance_info_for_left_out['performance'][f'train_{metric_name}'][split_idx] = train_metric
                    performance_info_for_left_out['performance'][f'eval_{metric_name}'][split_idx] = eval_metric
                    performance_info_for_left_out['performance'][f'test_{metric_name}'][split_idx] = test_metric

            logger.info('--- Average performance ---')
            performance_log = list()
            performance_log.append(f'# revised: {int(training_labels.sum())}')
            performance_log.append(f'# cases: {all_case_ids.shape[0]}')

            longest_scorer_name = max(len(name) for name in scoring.keys())
            for metric_name in scoring.keys():
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    try:
                        train_metric = np.nanmean(performance_info_for_left_out['performance'][f'train_{metric_name}'])
                        if np.isnan(train_metric):
                            train_metric = 0
                    except:
                        train_metric = 0

                    try:
                        eval_metric = np.nanmean(performance_info_for_left_out['performance'][f'eval_{metric_name}'])
                        if np.isnan(eval_metric):
                            eval_metric = 0
                    except:
                        eval_metric = 0

                    try:
                        test_metric = np.nanmean(performance_info_for_left_out['performance'][f'test_{metric_name}'])
                        if np.isnan(test_metric):
                            test_metric = 0
                    except:
                        test_metric = 0

                pad = ' ' * (longest_scorer_name - len(metric_name) + 1)

                msg = f'{metric_name}:{pad}train {train_metric:.6f}, eval {eval_metric:.6f}, test {test_metric:.6f}'
                performance_log.append(msg)
                logger.info(msg)

            os.makedirs(results_dir, exist_ok=True)
            with open(os.path.join(results_dir, 'performance.txt'), 'w') as f:
                f.writelines('\n'.join(performance_log))

            f_importances = list()
            for model in cv_models:
                f_importances.append(model.feature_importances_)
            f_importances = np.vstack(f_importances)
            mean_feature = np.mean(np.vstack(f_importances), axis=0)
            std_feature = np.std(np.vstack(f_importances), axis=0)

            pd.DataFrame({
                'feature': feature_names,
                'feature_importance_mean': mean_feature,
                'feature_importance_std': std_feature
            }).sort_values(by='feature_importance_mean', ascending=False).to_csv(
                    os.path.join(results_dir, 'feature_importances.csv'), index=False)

            all_performance_info_for_left_out.append(performance_info_for_left_out)

            logger.info(f"Storing predictions for '{LEAVE_ONE_OUT[0]}' in {LEAVE_ONE_OUT[1]} ...")
            results_dir_test_predictions = os.path.join(ROOT_DIR, 'results', folder_name, 'TEST_PREDICTIONS')
            if not os.path.exists(results_dir_test_predictions):
                os.makedirs(results_dir_test_predictions, exist_ok=True)
            filename_output = os.path.join(results_dir_test_predictions, f'{filename}-{LEAVE_ONE_OUT[0]}-{LEAVE_ONE_OUT[1]}.csv')

            test_scores_df = test_predictions_df.copy()
            test_scores_df['p'] = np.max(all_test_predictions, axis=1)
            test_classification_scores_df = test_scores_df.groupby('case_id').agg({
                'p': lambda x: np.max(x)
            }).reset_index(drop=False)
            case_ids = test_classification_scores_df['case_id'].values
            predictions = test_classification_scores_df['p'].values
            create_predictions_output_performance_app(filename_output, case_ids, predictions)


# noinspection PyTypeChecker
@beartype
def select_best_models(*, folder: str, metric: str, keep_top_n_prediction_files: int):
    p = Path(folder).glob('**/performance.txt')
    performance_files = [x for x in p if x.is_file()]

    logger.info(f'Found {len(performance_files)} performance files in {folder}')

    performance_table = list()
    for path in performance_files:
        with open(path, mode='r') as f:
            content = f.readlines()

        line = [l for l in content if l.startswith(metric)]
        if len(line) != 1:
            logger.warning(f'Unknown format of performance file {path}')
            continue

        performance_parts = line[0].split(':')[1].split(',')
        performance_parts = [p.strip() for p in performance_parts]
        eval_performance = [p for p in performance_parts if p.startswith('eval ')]
        if len(eval_performance) != 1:
            logger.warning(f'Unknown eval performance in {path}')
            continue

        eval_performance = float(eval_performance[0].split(' ')[1])
        model_name = Path(path).relative_to(folder).parent.name
        performance_table.append((model_name, eval_performance))

    performance_table = pd.DataFrame(performance_table, columns=['model_name', 'performance']).sort_values('performance', ascending=False).reset_index(drop=True)
    print(performance_table[:10])

    if keep_top_n_prediction_files > 0:
        folder_name_parts = Path(folder).name.split('_')
        hospital_name = folder_name_parts[1]
        year = folder_name_parts[2]

        for index, row in performance_table.iterrows():
            if int(index) >= keep_top_n_prediction_files:
                prediction_file = os.path.join(folder, 'TEST_PREDICTIONS', f"{row['model_name']}-{hospital_name}-{year}.csv")
                if os.path.exists(prediction_file):
                    os.remove(prediction_file)


if __name__ == '__main__':
    # train_random_forest(model_to_train='xgboost', overwrite=True)
    train_random_forest(model_to_train='ranker', overwrite=True)
    # train_random_forest(model_to_train='RF', overwrite=True)

    results_dir = os.path.join(ROOT_DIR, 'results', 'mind_bend_reviewed', 'test_KSW_2020')
    select_best_models(folder=results_dir, metric='NDCG', keep_top_n_prediction_files=1)

    create_rankings_of_revised_cases(
        filename_revised_cases=os.path.join(ROOT_DIR, "results/02_rf_hyperparameter_screen/01_runKSW_2020/ground_truth_performance_app_case_ranking_KSW_2020.csv"),
        dir_rankings=os.path.join(ROOT_DIR, 'results/mind_bend_reviewed/test_KSW_2020/TEST_PREDICTIONS/'),
        dir_output=os.path.join(ROOT_DIR, 'results/mind_bend_reviewed/test_KSW_2020/results/'),
        s3_bucket='code-scout'
    )
