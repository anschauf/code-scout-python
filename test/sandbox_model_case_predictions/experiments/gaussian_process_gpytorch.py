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
from loguru import logger
from sklearn.cluster import KMeans
from tqdm import tqdm

from src import ROOT_DIR
from test.sandbox_model_case_predictions.data_handler import load_data
from test.sandbox_model_case_predictions.experiments.GPModel import GPModel
from test.sandbox_model_case_predictions.experiments.PGLikelihood import PGLikelihood
from test.sandbox_model_case_predictions.experiments.StratifiedSampler import StratifiedSampler
from test.sandbox_model_case_predictions.utils import create_predictions_output_performance_app, \
    get_list_of_all_predictors, get_revised_case_ids, prepare_train_eval_test_split, \
    create_performance_app_ground_truth, sigmoid


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

        folder_name = f'test_GP_pytorch_all_features_{LEAVE_ON_OUT[0]}_{LEAVE_ON_OUT[1]}'
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
        DISCARDED_FEATURES = ('hospital', 'month_admission', 'month_discharge', 'year_discharge', 'mdc_OHE', 'hauptkostenstelle_OHE')
        all_data = load_data(only_2_rows=True)
        features_dir = join(ROOT_DIR, 'resources', 'features')
        feature_filenames, encoders = get_list_of_all_predictors(all_data, features_dir, overwrite=False)
        feature_names = sorted(list(feature_filenames.keys()))

        # feature_names = [feature_name for feature_name in feature_names
        #                  if not np.logical_or(np.logical_or(any(feature_name.startswith(discarded_feature) for discarded_feature in DISCARDED_FEATURES), feature_name.endswith('_OHE')), feature_name.endswith('_flag_RAW'))]
        feature_names = [feature_name for feature_name in feature_names
                         if not any(feature_name.startswith(discarded_feature) for discarded_feature in DISCARDED_FEATURES)]


        revised_cases_in_data = get_revised_case_ids(all_data, REVISED_CASE_IDS_FILENAME, overwrite=False)
        dir_ground_truth = join(ROOT_DIR, 'results', folder_name)
        create_performance_app_ground_truth(dir_ground_truth, revised_cases_in_data, LEAVE_ON_OUT[0], LEAVE_ON_OUT[1])
        # create_performance_app_ground_truth(dir_output, revised_cases_in_data, hospital_year_for_performance_app[0], hospital_year_for_performance_app[1])

        ind_train, ind_test, y_train, y_test, ind_hospital_leave_out, y_hospital_leave_out = \
            prepare_train_eval_test_split(dir_output=RESULTS_DIR, revised_cases_in_data=revised_cases_in_data,
                                          hospital_leave_out=LEAVE_ON_OUT[0],
                                          year_leave_out=LEAVE_ON_OUT[1],
                                          only_reviewed_cases=False)

        # reviewed_cases = revised_cases_in_data[(revised_cases_in_data['is_revised'] == 1) | (revised_cases_in_data['is_reviewed'] == 1)]
        reviewed_cases = revised_cases_in_data
        y = reviewed_cases['is_revised'].values
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


            from torch.utils.data import TensorDataset, DataLoader
            train_x = torch.Tensor(features[ind_train])
            train_y = torch.Tensor(y[ind_train])


            train_dataset = TensorDataset(train_x, train_y)
            batch_size = 2048
            stratified_sampler = StratifiedSampler(class_vector=torch.from_numpy(y[ind_train]), batch_size=batch_size)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=stratified_sampler)
            # train_loader = StratifiedSampler(train_dataset, batch_size=2048)

            # # we initialize our model with M = 30 inducing points
            M = 100
            # inducing_points = torch.linspace(-2., 2., M, dtype=train_x.dtype, device=train_x.device).unsqueeze(-1)
            # inducing_points = torch.Tensor(features[ind_train[:M]])
            # inducing_points = torch.Tensor(np.linspace(np.min(features[ind_train], axis=0),np.max(features[ind_train], axis=0), M))
            kmeans = KMeans(n_clusters=M, random_state=0, n_init="auto").fit(features[ind_train])
            inducing_points = torch.Tensor(kmeans.cluster_centers_)

            model = GPModel(inducing_points=inducing_points)
            # model.covar_module.base_kernel.initialize(lengthscale=0.2)
            model.covar_module.base_kernel.initialize()
            likelihood = PGLikelihood()

            if torch.cuda.is_available():
                model = model.cuda()
                likelihood = likelihood.cuda()

            variational_ngd_optimizer = gpytorch.optim.NGD(model.variational_parameters(), num_data=train_y.size(0), lr=0.01)

            hyperparameter_optimizer = torch.optim.Adam([
                {'params': model.hyperparameters()},
                {'params': likelihood.parameters()},
            ], lr=0.01)

            model.train()
            likelihood.train()
            mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

            smoke_test = ('CI' in os.environ)
            num_epochs = 1 if smoke_test else 100
            epochs_iter = tqdm(range(num_epochs), desc="Epoch")
            loss_values = list()
            for i in epochs_iter:
                minibatch_iter = tqdm(train_loader, desc="Minibatch", leave=False)

                epoch_iteration = list()
                for x_batch, y_batch in minibatch_iter:
                    ### Perform NGD step to optimize variational parameters
                    variational_ngd_optimizer.zero_grad()
                    hyperparameter_optimizer.zero_grad()

                    output = model(x_batch)
                    loss = -mll(output, y_batch)
                    epoch_iteration.append(loss.detach().numpy())
                    minibatch_iter.set_postfix(loss=loss.item())
                    loss.backward()
                    variational_ngd_optimizer.step()
                    hyperparameter_optimizer.step()
                loss_values.append(np.mean(epoch_iteration))

                plt.figure()
                plt.plot(range(len(loss_values)), loss_values)
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.savefig(join(RESULTS_DIR, 'loss.pdf'), bbox_inches='tight')
                plt.close()

            model_preds_train = model(torch.Tensor(train_x))
            prediction_logits_train = model_preds_train.loc.data.cpu().detach().numpy()
            prediction_variance_train = model_preds_train.variance.detach().numpy()
            predictions_train = np.asarray([sigmoid(x) for x in prediction_logits_train])
            predictions_confidence_region_lower_train, predictions_confidence_region_upper_train = model_preds_train.confidence_region()
            predictions_confidence_region_lower_train = np.asarray([sigmoid(x) for x in predictions_confidence_region_lower_train.detach().numpy()])
            predictions_confidence_region_upper_train = np.asarray([sigmoid(x) for x in predictions_confidence_region_upper_train.detach().numpy()])
            confidence_interval_size_train = predictions_confidence_region_upper_train - predictions_confidence_region_lower_train

            add_on_information = pd.DataFrame({
                'predictions': predictions_train,
                'predictions_confidence_region_lower': predictions_confidence_region_lower_train,
                'predictions_confidence_region_upper_train': predictions_confidence_region_upper_train,
                'confidence_interval_size': confidence_interval_size_train,
                'variance': prediction_variance_train,
                'y_label': y_train
            })

            plt.figure()
            sns.histplot(data=add_on_information, x='confidence_interval_size', hue='y_label', kde=True, common_norm=False)
            plt.savefig(join(RESULTS_DIR, 'hist_confidence_interval_size_train.pdf'), bbox_inches='tight', )
            plt.close()

            plt.figure()
            sns.histplot(data=add_on_information[add_on_information['y_label'] == 1], x='confidence_interval_size', hue='y_label', kde=True, common_norm=False)
            plt.savefig(join(RESULTS_DIR, 'hist_confidence_interval_size_train_revised.pdf'), bbox_inches='tight', )
            plt.close()

            predictions_df = pd.DataFrame({'predictions': predictions_train, 'y_label': y_train})
            plt.figure()
            sns.histplot(data=predictions_df, x='predictions', hue='y_label', kde=True, common_norm=False)
            plt.savefig(join(RESULTS_DIR, 'hist_predictions_train.pdf'), bbox_inches='tight', )
            plt.close()

            predictions_df = pd.DataFrame({'predictions': predictions_train + predictions_confidence_region_upper_train, 'y_label': y_train})
            plt.figure()
            sns.histplot(data=predictions_df, x='predictions', hue='y_label', kde=True, common_norm=False)
            plt.savefig(join(RESULTS_DIR, 'hist_predictions_plus_upper_confidence_region_train.pdf'), bbox_inches='tight', )
            plt.close()


            logger.info('Training finished!')



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

                model_preds = model(torch.Tensor(test_features))
                prediction_logits = model_preds.loc.data.cpu().detach().numpy()
                prediction_variance = model_preds.variance.detach().numpy()
                predictions = np.asarray([sigmoid(x) for x in prediction_logits])
                predictions_confidence_region_lower, predictions_confidence_region_upper =  model_preds.confidence_region()
                predictions_confidence_region_lower = np.asarray([sigmoid(x) for x in predictions_confidence_region_lower.detach().numpy()])
                predictions_confidence_region_upper = np.asarray([sigmoid(x) for x in predictions_confidence_region_upper.detach().numpy()])
                confidence_interval_size = predictions_confidence_region_upper-predictions_confidence_region_lower

                filename_output = join(RESULTS_DIR_TEST_PREDICTIONS, f'gaussian_process_{LEAVE_ON_OUT[0]}-{LEAVE_ON_OUT[1]}.csv')


                add_on_information = pd.DataFrame({
                    'confidence_interval_size': confidence_interval_size,
                    'variance': prediction_variance,
                    'y_label': hospital_data['is_revised'].values
                })

                plt.figure()
                sns.histplot(data=add_on_information, x='confidence_interval_size', hue='y_label', kde=True, common_norm=False)
                plt.savefig(join(RESULTS_DIR, 'hist_confidence_interval_size.pdf'), bbox_inches='tight',)
                plt.close()

                plt.figure()
                sns.histplot(data=add_on_information[add_on_information['y_label'] == 1], x='confidence_interval_size', hue='y_label', kde=True, common_norm=False)
                plt.savefig(join(RESULTS_DIR, 'hist_confidence_interval_size_revised.pdf'), bbox_inches='tight',)
                plt.close()


                predictions_df = pd.DataFrame({'predictions': predictions, 'y_label': hospital_data['is_revised'].values})
                plt.figure()
                sns.histplot(data=predictions_df[predictions_df['y_label'] == 1], x='predictions', hue='y_label', kde=True, common_norm=False)
                plt.savefig(join(RESULTS_DIR, 'hist_predictions_revised.pdf'), bbox_inches='tight',)
                plt.close()


                create_predictions_output_performance_app(filename=filename_output,
                                                          case_ids=case_ids,
                                                          predictions=predictions,
                                                          add_on_information=add_on_information)


                predictions_variance_corrected = np.copy(predictions)
                # # predictions_variance_corrected[prediction_variance < 0.2] = 0
                # predictions_variance_corrected[confidence_interval_size < 0.1] = 0
                predictions_variance_corrected += predictions_confidence_region_upper
                predictions_variance_corrected[predictions < 0.05] = 0
                create_predictions_output_performance_app(filename=filename_output.replace('.csv', '_variance_filtered.csv'),
                                                          case_ids=case_ids,
                                                          predictions=predictions_variance_corrected,
                                                          add_on_information=add_on_information)


    logger.success('done')


if __name__ == '__main__':
    train_random_forest_only_reviewed_cases()
    sys.exit(0)
