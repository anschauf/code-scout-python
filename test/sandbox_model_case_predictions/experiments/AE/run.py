import json
import os
import sys
import warnings
from os.path import join

import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from src import ROOT_DIR
from test.sandbox_model_case_predictions.data_handler import load_data
from test.sandbox_model_case_predictions.experiments.AE.ae_model import Autoencoder
from test.sandbox_model_case_predictions.experiments.AE.balanced_generator import BalancedGenerator
from test.sandbox_model_case_predictions.utils import create_predictions_output_performance_app, \
    get_list_of_all_predictors, get_revised_case_ids, prepare_train_eval_test_split, \
    create_performance_app_ground_truth


def train_ae():

    parameters = dict()
    parameters['dim_latent_non_categorical'] = 6
    parameters['dim_latent_categorical'] = 6
    parameters['dim_latent_count'] = 6
    parameters['batch_size'] = 128
    parameters['epochs'] = 200


    # for LEAVE_ON_OUT in [
    #     ('AA', 2017), ('AA', 2018), ('AA', 2019), ('AA', 2020), ('AA', 2021),
    #     ('AK', 2017), ('AK', 2018), ('AK', 2019), ('AK', 2020),
    #     ('BC', 2017), ('BC', 2018), ('BC', 2019), ('BC', 2020),
    #     ('BE', 2017), ('BE', 2018), ('BE', 2019),
    #     ('BI', 2017), ('BI', 2018), ('BI', 2019), ('BI', 2020), ('BI', 2021),
    #     ('BS', 2017), ('BS', 2018), ('BS', 2019), ('BS', 2020), ('BS', 2021),
    #     ('CC', 2017), ('CC', 2018), ('CC', 2019),
    #     ('FT', 2017), ('FT', 2018), ('FT', 2019),
    #     ('HI', 2016), ('HI', 2017), ('HI', 2018), ('HI', 2019), ('HI', 2020), ('HI', 2021),
    #     ('IP', 2017), ('IP', 2018), ('IP', 2019), ('IP', 2020), ('IP', 2021),
    #     ('KSSG', 2019), ('KSSG', 2020), ('KSSG', 2021),
    #     ('KSW', 2015), ('KSW', 2017), ('KSW', 2018), ('KSW', 2019), ('KSW', 2020),
    #     ('LC', 2017), ('LC', 2018), ('LC', 2019),
    #     ('LI', 2017), ('LI', 2018), ('LI', 2019), ('LI', 2020), ('LI', 2021),
    #     ('MG', 2017), ('MG', 2018), ('MG', 2019),
    #     ('PM', 2017), ('PM', 2018), ('PM', 2019),
    #     ('RO', 2017), ('RO', 2018), ('RO', 2019),
    #     ('SA', 2015), ('SA', 2016), ('SA', 2017), ('SA', 2018), ('SA', 2019), ('SA', 2020),
    #     ('SH', 2017), ('SH', 2018), ('SH', 2019), ('SH', 2020),
    #     ('SLI', 2017), ('SLI', 2018), ('SLI', 2019),
    #     ('SRRWS', 2017), ('SRRWS', 2018), ('SRRWS', 2019),
    #     ('ST', 2017), ('ST', 2018), ('ST', 2019), ('ST', 2020),
    #     ('USZ', 2019)
    # ]:

    for LEAVE_ON_OUT in [
        # ('FT', 2019),
        # ('HI', 2016),
        # ('KSSG', 2021),
        # ('KSW', 2017), ('KSW', 2018),
        # ('KSW', 2019),
        ('KSW', 2020),
        # ('LI', 2017), ('LI', 2018),
        # ('SLI', 2019),
        # ('SRRWS', 2019)
    ]:

        folder_name = f'01_autoencoder/{LEAVE_ON_OUT[0]}_{LEAVE_ON_OUT[1]}'
        RESULTS_DIR = join(ROOT_DIR, 'results', folder_name)
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
            json.dump(parameters, open(join(RESULTS_DIR, "parameters.txt"), 'w'))

            RESULTS_DIR_TEST_PREDICTIONS = join(ROOT_DIR, 'results', folder_name, 'TEST_PREDICTIONS')
            if not os.path.exists(RESULTS_DIR_TEST_PREDICTIONS):
                os.makedirs(RESULTS_DIR_TEST_PREDICTIONS)

            REVISED_CASE_IDS_FILENAME = join(ROOT_DIR, 'resources', 'data', 'revised_case_ids.csv')

            DISCARDED_FEATURES = ('hospital', 'month_admission', 'month_discharge', 'year_discharge', 'mdc_OHE', 'hauptkostenstelle_OHE', 'gender_RAW'
                                  # 'behandlung_nach_austritt_OHE', 'is_case_below_pccl_split_RAW', 'duration_of_stay_case_type_OHE', 'binned_age_RAW', 'aufenthalts_klasse_OHE'
                                  )

            all_data = load_data(only_2_rows=True)
            features_dir = join(ROOT_DIR, 'resources', 'features')
            feature_filenames, encoders = get_list_of_all_predictors(all_data, features_dir, overwrite=False)
            feature_names = sorted(list(feature_filenames.keys()))

            feature_names = [feature_name for feature_name in feature_names
                             if not any(feature_name.startswith(discarded_feature) for discarded_feature in DISCARDED_FEATURES)]

            revised_cases_in_data = get_revised_case_ids(all_data, REVISED_CASE_IDS_FILENAME, overwrite=False)
            dir_ground_truth = join(ROOT_DIR, 'results', folder_name)
            create_performance_app_ground_truth(dir_ground_truth, revised_cases_in_data, LEAVE_ON_OUT[0], LEAVE_ON_OUT[1])
            # create_performance_app_ground_truth(dir_output, revised_cases_in_data, hospital_year_for_performance_app[0], hospital_year_for_performance_app[1])

            ind_train, ind_test, y_train, y_val, ind_hospital_leave_out, y_hospital_leave_out = \
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

            ind_categorical_features = [i for i in range(len(feature_ids)) if ('_flag_RAW' in feature_ids[i]
                                                                               or '_ohe_' in feature_ids[i].lower()
                                                                               or feature_ids[i].startswith('binned_age_')
                                                                               or 'has_complex_procedure_RAW' in feature_ids[i]
                                                                               or 'has_ventilation_hours_AND_in_pre_mdc_RAW' in feature_ids[i]
                                                                               or 'has_ventilation_hours_RAW' in feature_ids[i]
                                                                               or 'is_case_below_pccl_split_RAW' in feature_ids[i]
                                                                               or 'is_emergency_case_RAW' in feature_ids[i])]
            ind_non_categorical_features = list(set(list(range(len(feature_ids)))) - set(ind_categorical_features))

            ind_count_features = [i for i in ind_non_categorical_features if (
                        'pccl_RAW' == feature_ids[i]
                      or 'age_years_RAW' == feature_ids[i]
                      or 'number_of_diags_RAW' == feature_ids[i]
                      or 'number_of_diags_ccl_greater_0_RAW' == feature_ids[i]
                      or 'number_of_chops_RAW' == feature_ids[i]
                      or 'num_drg_relevant_diagnoses_RAW' == feature_ids[i]
                      or 'num_drg_relevant_procedures_RAW' == feature_ids[i]
                      )]
            ind_non_categorical_features = list(set(list((ind_non_categorical_features))) - set(ind_count_features))


            features = np.hstack(features)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')

                # train AE here
                ae = Autoencoder(RESULTS_DIR, 
                                 dim_latent_non_categorical=parameters['dim_latent_non_categorical'], 
                                 dim_latent_categorical=parameters['dim_latent_categorical'],
                                 dim_latent_count=parameters['dim_latent_count'],
                                 dim_input_categorical=len(ind_categorical_features),
                                 dim_input_non_categorical=len(ind_non_categorical_features), 
                                 dim_input_count=len(ind_count_features))
                ae.model_train.summary()
                # input == output with an autoencoder

                features_train = features[ind_train]
                y_train = y[ind_train]

                # scale non categorical features
                scaler = MinMaxScaler()
                scaler.fit(features_train[:, ind_non_categorical_features])
                features_train_non_categorical_scaled = scaler.transform(features_train[:, ind_non_categorical_features])

                data_train = [features_train[:, ind_categorical_features], features_train_non_categorical_scaled, features_train[:, ind_count_features]]
                features_val = features[ind_test]
                y_val = y[ind_test]

                features_val_non_categorical_scaled = scaler.transform(features_val[:, ind_non_categorical_features])
                data_val = [features_val[:, ind_categorical_features], features_val_non_categorical_scaled, features_val[:, ind_count_features], y_val]
                data_val_predict = [features_val[:, ind_categorical_features], features_val_non_categorical_scaled, features_val[:, ind_count_features]]

                balanced_generator = BalancedGenerator(parameters['batch_size'],
                                                       features_train[:, ind_categorical_features],
                                                       features_train_non_categorical_scaled,
                                                       features_train[:, ind_count_features], y_train)
                history = ae.model_train.fit_generator(balanced_generator, verbose=True, epochs=parameters['epochs'],
                                                       validation_data=(data_val, data_val))

                ae.model_train.save_weights(os.path.join(RESULTS_DIR, './checkpoints/my_checkpoint'))

                latent_space_train = ae.model_latent_space.predict(data_train)
                latent_space_val = ae.model_latent_space.predict(data_val_predict)
                predictions_revision_train = ae.model_predict_revision.predict(data_train)
                predictions_revision_train_binary = [1 if (x > .5) else 0 for x in predictions_revision_train]
                predictions_revision_val = ae.model_predict_revision.predict(data_val_predict)
                predictions_revision_val_binary = [1 if (x > .5) else 0 for x in predictions_revision_val]

                f1_train = f1_score(y_train, predictions_revision_train_binary)
                f1_val = f1_score(y_val, predictions_revision_val_binary)
                precision_train = precision_score(y_train, predictions_revision_train_binary)
                precision_val = precision_score(y_val, predictions_revision_val_binary)
                recall_train = recall_score(y_train, predictions_revision_train_binary)
                recall_val = recall_score(y_val, predictions_revision_val_binary)

                logger.info('plotting loss functions')
                loss_types = np.unique(['_'.join(x.split('_')[:-1]).replace('val_', '') if not x in ['loss', 'val_loss'] else 'loss' for x in history.history.keys()])
                for loss_type in loss_types:
                    plt.figure()
                    for key in history.history.keys():
                        if key.startswith(loss_type) or key.startswith('val_'+ loss_type):
                            plt.plot(history.history[key], label=key)
                    plt.title(loss_type)
                    plt.ylabel('loss')
                    plt.xlabel('epoch')
                    plt.legend(loc='best')
                    plt.savefig(os.path.join(RESULTS_DIR, f'loss_{loss_type}.pdf'), bbox_inches='tight')
                    plt.close()

                pca = PCA(n_components=3)
                pca.fit(latent_space_train)
                latent_space_train_pca = pca.transform(latent_space_train)
                latent_space_test_pca = pca.transform(latent_space_val)

                plt.figure()
                ind_revised = np.where(y_train == 1)[0]
                ind_not_revised = np.where(y_train == 0)[0]
                plt.scatter(latent_space_train_pca[ind_not_revised][:, 0], latent_space_train_pca[ind_not_revised][:, 1], c='b', alpha=0.5, s=1)
                plt.scatter(latent_space_train_pca[ind_revised][:, 0], latent_space_train_pca[ind_revised][:, 1], c='r', s=1)
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.savefig(join(RESULTS_DIR, 'pca_latent_space_train.pdf'), bbox_inches='tight')
                plt.close()

                plt.figure()
                ind_revised = np.where(y_val == 1)[0]
                ind_not_revised = np.where(y_val == 0)[0]
                plt.scatter(latent_space_test_pca[ind_not_revised][:, 0], latent_space_test_pca[ind_not_revised][:, 1], c='b', alpha=0.5, s=1)
                plt.scatter(latent_space_test_pca[ind_revised][:, 0], latent_space_test_pca[ind_revised][:, 1], c='r', s=1)
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.savefig(join(RESULTS_DIR, 'pca_latent_space_val.pdf'), bbox_inches='tight')
                plt.close()


                n_top = 1000
                ##### categorical reconstruction
                feature_ids_categorical = np.asarray(feature_ids)[ind_categorical_features]
                original_categorical = features_train[:n_top][:, ind_categorical_features]
                pd.DataFrame(original_categorical, columns=feature_ids_categorical).to_csv(join(RESULTS_DIR, 'original_categorical.csv'), index=False)
                reconstructed_categorical = ae.model_reconstruction_categorical.predict(original_categorical)
                pd.DataFrame(reconstructed_categorical, columns=feature_ids_categorical).to_csv(join(RESULTS_DIR, 'reconstructed_categorical.csv'), index=False)
                feature_reconstruction_categorical = pd.DataFrame(original_categorical - reconstructed_categorical, columns=feature_ids_categorical)
                plt.figure()
                cluster_categorical = sns.clustermap(feature_reconstruction_categorical)
                plt.savefig(join(RESULTS_DIR, 'clustermap_feature_reconstruction_categorical.pdf'), bbox_inches='tight')
                plt.close()
                pd.DataFrame({
                    'reordered_features': np.asarray(feature_ids)[ind_categorical_features][cluster_categorical.dendrogram_col.reordered_ind],
                    'mean_distance':np.mean(feature_reconstruction_categorical.values, axis=0)[cluster_categorical.dendrogram_col.reordered_ind]
                }).sort_values(by='mean_distance').to_csv(join(RESULTS_DIR, 'clustermap_reordered_features_average_error_categorical.csv'), index=False)


                ##### non categorical reconstruction
                feature_ids_non_categorical = np.asarray(feature_ids)[ind_non_categorical_features]
                original_non_categorical = features_train_non_categorical_scaled[:n_top]
                pd.DataFrame(original_non_categorical, columns=feature_ids_non_categorical).to_csv(join(RESULTS_DIR, 'original_non_categorical.csv'), index=False)
                reconstructed_non_categorical = ae.model_reconstruction_non_categorical.predict(original_non_categorical)
                pd.DataFrame(reconstructed_non_categorical, columns=feature_ids_non_categorical).to_csv(join(RESULTS_DIR, 'reconstructed_non_categorical.csv'), index=False)
                feature_reconstruction_non_categorical = pd.DataFrame(original_non_categorical - reconstructed_non_categorical, columns=feature_ids_non_categorical)
                plt.figure()
                cluster_non_categorical = sns.clustermap(feature_reconstruction_non_categorical.dropna(), vmin=-1, vmax=1)
                plt.savefig(join(RESULTS_DIR, 'clustermap_feature_reconstruction_non_categorical.pdf'), bbox_inches='tight')
                plt.close()
                pd.DataFrame({
                    'reordered_features': np.asarray(feature_ids)[ind_non_categorical_features][cluster_non_categorical.dendrogram_col.reordered_ind],
                    'mean_distance':np.mean(feature_reconstruction_non_categorical.values, axis=0)[cluster_non_categorical.dendrogram_col.reordered_ind]
                }).sort_values(by='mean_distance').to_csv(join(RESULTS_DIR, 'clustermap_reordered_features_average_error_non_categorical.csv'), index=False)


                ##### count reconstruction
                feature_ids_count = np.asarray(feature_ids)[ind_count_features]
                original_count = features_train[:n_top][:, ind_count_features]
                pd.DataFrame(original_count, columns=feature_ids_count).to_csv(join(RESULTS_DIR, 'original_count.csv'), index=False)
                reconstructed_count = ae.model_reconstruction_count.predict(features_train[:n_top][:, ind_count_features])
                pd.DataFrame(reconstructed_count, columns=feature_ids_count).to_csv(join(RESULTS_DIR, 'reconstructed_count.csv'), index=False)
                feature_reconstruction_count = pd.DataFrame(original_count - reconstructed_count, columns=feature_ids_count)
                plt.figure()
                cluster_count = sns.clustermap(feature_reconstruction_count)
                plt.savefig(join(RESULTS_DIR, 'clustermap_feature_reconstruction_count.pdf'), bbox_inches='tight')
                plt.close()
                pd.DataFrame({
                    'reordered_features': np.asarray(feature_ids)[ind_count_features][cluster_count.dendrogram_col.reordered_ind],
                    'mean_distance':np.mean(feature_reconstruction_count.values, axis=0)[cluster_count.dendrogram_col.reordered_ind]
                }).sort_values(by='mean_distance').to_csv(join(RESULTS_DIR, 'clustermap_reordered_features_average_error_count.csv'), index=False)



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

                    test_features_non_categorical_scaled = scaler.transform(test_features[:, ind_non_categorical_features])
                    data_test = [test_features[:, ind_categorical_features], test_features_non_categorical_scaled, test_features[:, ind_count_features]]
                    predictions_loo = ae.model_predict_revision.predict(data_test).flatten()

                    filename_output = join(RESULTS_DIR_TEST_PREDICTIONS, f'ae_model.csv')

                    create_predictions_output_performance_app(filename=filename_output,
                                                              case_ids=case_ids,
                                                              predictions=predictions_loo)

            logger.info(f'{f1_train=}, {f1_val=} | {recall_train=}, {recall_val=} | {precision_train=}, {precision_val=}')
    logger.success('done')


if __name__ == '__main__':
    train_ae()
    sys.exit(0)
