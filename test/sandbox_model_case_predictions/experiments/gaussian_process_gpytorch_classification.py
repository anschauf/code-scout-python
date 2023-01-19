import os
import sys
import warnings
from os.path import join

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from gpytorch.mlls import VariationalELBO
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, recall_score, precision_score
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from src import ROOT_DIR
from test.sandbox_model_case_predictions.data_handler import load_data
from test.sandbox_model_case_predictions.experiments.GPModelClassification import GPClassificationModel
from test.sandbox_model_case_predictions.experiments.StratifiedSampler import StratifiedSampler
from test.sandbox_model_case_predictions.utils import get_list_of_all_predictors, get_revised_case_ids, \
    prepare_train_eval_test_split, \
    create_performance_app_ground_truth, create_predictions_output_performance_app


def train_random_forest_only_reviewed_cases():

    for LEAVE_ON_OUT in [
        # ('FT', 2019),
        # ('HI', 2016),
        # ('KSSG', 2021),
        # ('KSW', 2017), ('KSW', 2018), ('KSW', 2019),
        ('KSW', 2020),
        # ('LI', 2017), ('LI', 2018),
        # ('SLI', 2019),
        # ('SRRWS', 2019),
        # ('USZ', 2019)
    ]:

        folder_name = f'test_GP_pytorch_all_features_test_{LEAVE_ON_OUT[0]}_{LEAVE_ON_OUT[1]}'
        RESULTS_DIR = join(ROOT_DIR, 'results', folder_name,
                           f'cross_validation')
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)

        RESULTS_DIR_TEST_PREDICTIONS = join(ROOT_DIR, 'results', folder_name, 'TEST_PREDICTIONS')
        if not os.path.exists(RESULTS_DIR_TEST_PREDICTIONS):
            os.makedirs(RESULTS_DIR_TEST_PREDICTIONS)

        REVISED_CASE_IDS_FILENAME = join(ROOT_DIR, 'resources', 'data', 'revised_case_ids.csv')

        # DISCARDED_FEATURES = ('hospital', 'month_admission', 'month_discharge', 'year_discharge', 'mdc_OHE', 'hauptkostenstelle_OHE', 'binned_age_RAW', 'aufenthalt_intensivstation_RAW', 'has_complex_procedure_RAW',
        #                       'has_ventilation_hours_AND_in_pre_mdc_RAW', 'has_ventilation_hours_RAW', 'is_case_below_pccl_split_RAW', 'is_emergency_case_RAW')
        # DISCARDED_FEATURES = ('hospital', 'month_admission', 'month_discharge', 'year_discharge', 'mdc_OHE', 'hauptkostenstelle_OHE')
        SELECTED_FEATURES = ['age_years_RAW', 'effective_cost_weight_RAW', 'duration_of_stay_RAW', 'num_drg_relevant_procedures_RAW', 'number_of_diags_ccl_greater_0_RAW', 'raw_pccl_RAW', 'delta_ccl_to_next_pccl_RAW', 'aufenthalt_intensivstation_RAW', 'delta_effective_to_base_drg_cost_weight_RAW',
                             'erfassung_der_aufwandpunkte_fuer_imc_RAW']
        all_data = load_data(only_2_rows=True)
        features_dir = join(ROOT_DIR, 'resources', 'features')
        feature_filenames, encoders = get_list_of_all_predictors(all_data, features_dir, overwrite=False)
        feature_names = sorted(list(feature_filenames.keys()))

        # feature_names = [feature_name for feature_name in feature_names
        #                  if not np.logical_or(np.logical_or(any(feature_name.startswith(discarded_feature) for discarded_feature in DISCARDED_FEATURES), feature_name.endswith('_OHE')), feature_name.endswith('_flag_RAW'))]
        feature_names = [feature_name for feature_name in feature_names
                         if any(feature_name.startswith(selected_feature) for selected_feature in SELECTED_FEATURES)]
        # feature_names = [feature_name for feature_name in feature_names
        #                  if not any(feature_name.startswith(discarded_feature) for discarded_feature in DISCARDED_FEATURES)]


        revised_cases_in_data = get_revised_case_ids(all_data, REVISED_CASE_IDS_FILENAME, overwrite=False)
        dir_ground_truth = join(ROOT_DIR, 'results', folder_name)
        create_performance_app_ground_truth(dir_ground_truth, revised_cases_in_data, LEAVE_ON_OUT[0], LEAVE_ON_OUT[1])
        # create_performance_app_ground_truth(dir_output, revised_cases_in_data, hospital_year_for_performance_app[0], hospital_year_for_performance_app[1])

        ind_train, ind_test, y_train, y_test, ind_hospital_leave_out, y_hospital_leave_out = \
            prepare_train_eval_test_split(dir_output=RESULTS_DIR, revised_cases_in_data=revised_cases_in_data,
                                          hospital_leave_out=LEAVE_ON_OUT[0],
                                          year_leave_out=LEAVE_ON_OUT[1],
                                          only_reviewed_cases=True)

        # reviewed_cases = revised_cases_in_data[(revised_cases_in_data['is_revised'] == 1) | (revised_cases_in_data['is_reviewed'] == 1)]
        reviewed_cases = revised_cases_in_data
        y = reviewed_cases['is_revised'].values
        y[y == 0] = -1
        sample_indices = reviewed_cases['index'].values

        logger.info('Assembling features ...')
        features = list()
        feature_ids = list()
        for feature_name in feature_names:
            feature_filename = feature_filenames[feature_name]
            feature_values = np.load(feature_filename, mmap_mode='r', allow_pickle=False, fix_imports=False)
            features.append(feature_values[sample_indices, :])
            feature_ids.append([f'{feature_name}_{i}' for i in range(feature_values.shape[1])] if feature_values.shape[1] > 1 else [feature_name])

        feature_ids = np.concatenate(feature_ids)
        features = np.hstack(features)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            logger.info('Training the model..')

            train_x = torch.Tensor(features[ind_train])
            train_y = torch.Tensor(y[ind_train])
            test_x = torch.Tensor(features[ind_test])
            test_y = torch.Tensor(y[ind_test])

            train_dataset = TensorDataset(train_x, train_y)
            batch_size = 128
            stratified_sampler = StratifiedSampler(class_vector=torch.from_numpy(y[ind_train]), batch_size=batch_size)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=stratified_sampler)

            M = 125
            kmeans = KMeans(n_clusters=M, random_state=0, n_init="auto").fit(features[ind_train])
            inducing_points = torch.Tensor(kmeans.cluster_centers_)
            # ind_inducing_points = np.random.choice(ind_train, size=M, replace=False)
            # inducing_points = torch.Tensor(features[ind_inducing_points])

            model = GPClassificationModel(train_x=inducing_points)
            model.covar_module.base_kernel.initialize(lengthscale=2)
            # model.covar_module.base_kernel.initialize_from_data(train_x, train_y)
            likelihood = gpytorch.likelihoods.BernoulliLikelihood()

            # Find optimal model hyperparameters
            model.train()
            likelihood.train()

            # Use the adam optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

            # "Loss" for GPs - the marginal log likelihood
            # num_data refers to the amount of training data
            mll = VariationalELBO(likelihood, model, train_y.numel())

            training_iter = 300
            loss_values = list()
            for i in range(training_iter):
                minibatch_iter = tqdm(train_loader, desc="Minibatch", leave=False)

                epoch_iteration = list()
                for x_batch, y_batch in minibatch_iter:
                    # Zero backpropped gradients from previous iteration
                    optimizer.zero_grad()
                    # Get predictive output
                    output = model(x_batch)
                    # Calc loss and backprop gradients
                    loss = -mll(output, y_batch)
                    epoch_iteration.append(loss.detach().numpy())
                    loss.backward()
                    optimizer.step()
                loss_values.append(np.mean(epoch_iteration))
                print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss_values[-1]))

                plt.figure()
                plt.plot(range(len(loss_values)), loss_values)
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.savefig(join(RESULTS_DIR, 'loss.pdf'), bbox_inches='tight')
                plt.close()

            logger.info('Training finished!')

        # Go into eval mode
        model.eval()
        likelihood.eval()

        with torch.no_grad():


            f_preds_train = model(train_x)
            variance_train = f_preds_train.variance.detach().numpy()
            lower_train, upper_train = f_preds_train.confidence_region()
            y_preds_train = likelihood(model(train_x))
            y_preds_train_probas_detached = y_preds_train.probs.detach().numpy()
            y_preds_train_probas_detached_binary = [1 if x > 0.5 else 0 for x in y_preds_train_probas_detached]

            df_results_train = pd.DataFrame({
                'y_label': train_y,
                'probabilities': y_preds_train_probas_detached,
                'variance': variance_train,
                'lower': lower_train,
                'upper': upper_train
            })

            f_preds_test = model(test_x)
            variance_test = f_preds_test.variance.detach().numpy()
            lower_test, upper_test = f_preds_test.confidence_region()
            y_preds_test = likelihood(model(test_x))
            y_preds_test_probas_detached = y_preds_test.probs.detach().numpy()
            y_preds_test_probas_detached_binary = [1 if x > 0.5 else 0 for x in y_preds_test_probas_detached]

            df_results_test = pd.DataFrame({
                'y_label': test_y,
                'variance': variance_test,
                'probabilities': y_preds_test_probas_detached,
                'lower': lower_test,
                'upper': upper_test
            })

            def plot_barplots(df_results, tag):
                for hue in ['variance', 'probabilities', 'lower', 'upper']:
                    plt.figure()
                    sns.boxplot(data=df_results, x='y_label', y=hue)
                    plt.xlabel('y_label')
                    plt.ylabel(hue)
                    plt.savefig(join(RESULTS_DIR, f'boxplot_{tag}_{hue}_vs_y_label.pdf'), bbox_inches='tight')
                    plt.close()

            plot_barplots(df_results_train, 'train')
            plot_barplots(df_results_test, 'test')

            f1_test = f1_score(y_test, y_preds_test_probas_detached_binary)
            f1_train = f1_score(y_train, y_preds_train_probas_detached_binary)
            recall_test = recall_score(y_test, y_preds_test_probas_detached_binary)
            recall_train = recall_score(y_train, y_preds_train_probas_detached_binary)
            precision_test = precision_score(y_test, y_preds_test_probas_detached_binary)
            precision_train = precision_score(y_train, y_preds_train_probas_detached_binary)

            # plot lower, upper bound
            for i in range(features[ind_test].shape[1]):
                feature_values = features[ind_test][:, i]
                ind_arg_sort = np.argsort(feature_values)
                plt.figure()
                plt.fill_between(feature_values[ind_arg_sort], lower_test.numpy()[ind_arg_sort], upper_test.numpy()[ind_arg_sort], alpha=0.5)
                plt.xlabel(feature_ids[i])
                plt.ylabel('Confidence boundary')
                plt.ylim([-2, 2])
                plt.savefig(join(RESULTS_DIR, f'confidence_boundary_{feature_ids[i]}.pdf'))
                plt.close()


            logger.info('--- Average performance ---')
            performance_log = list()
            performance_log.append(f'# revised: {int(y.sum())}')
            performance_log.append(f'# cases: {y.shape[0]}')

            logger.info('Calculating and storing predictions for each combination of hospital and year, which contains revised cases ...')
            # List the hospitals and years for which there are revised cases
            all_hospitals_and_years = revised_cases_in_data[revised_cases_in_data['is_revised'] == 1][['hospital', 'dischargeYear']].drop_duplicates().values.tolist()
            for info in tqdm(all_hospitals_and_years):
                hospital_name = info[0]
                discharge_year = info[1]
                if hospital_name == LEAVE_ON_OUT[0] and discharge_year == LEAVE_ON_OUT[1]:
                    hospital_data = revised_cases_in_data[(revised_cases_in_data['hospital'] == hospital_name) & (revised_cases_in_data['dischargeYear'] == discharge_year)]

                    indices = hospital_data['index'].values
                    case_ids = hospital_data['id'].values

                    test_features = list()
                    for feature_name in feature_names:
                        feature_filename = feature_filenames[feature_name]
                        feature_values = np.load(feature_filename, mmap_mode='r', allow_pickle=False, fix_imports=False)
                        test_features.append(feature_values[indices, :])
                    test_features = np.hstack(test_features)

                    f_preds_test = model(torch.Tensor(test_features))
                    y_preds_test = likelihood(model(torch.Tensor(test_features)))
                    upper, lower = f_preds_test.confidence_region()

                    predictions = y_preds_test.probs.detach().numpy()

                    filename_output = join(RESULTS_DIR_TEST_PREDICTIONS, f'gaussian_process_{LEAVE_ON_OUT[0]}-{LEAVE_ON_OUT[1]}.csv')
                    create_predictions_output_performance_app(filename=filename_output,
                                                              case_ids=case_ids,
                                                              predictions=predictions,
                                                              add_on_information=None)




    logger.info(f'{precision_train=}, {precision_test=}, {recall_train=}, {recall_test=}, {f1_train=}, {f1_test=}')
    logger.success('done')


if __name__ == '__main__':
    train_random_forest_only_reviewed_cases()
    sys.exit(0)
