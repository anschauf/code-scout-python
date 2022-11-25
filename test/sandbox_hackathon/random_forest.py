from os import makedirs
from os.path import join, exists

import awswrangler as wr
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd

from src import PROJECT_ROOT_DIR
from test.sandbox_hackathon.constants import FILENAME_TRAIN_SPLIT, FILENAME_TEST_SPLIT, RANDOM_SEED
from test.sandbox_hackathon.feature_extraction_functions import extract_number_of, get_categorized_predictors, \
    get_continous_int_variable
from test.sandbox_hackathon.utils import load_data, train_rf_model, write_model_coefs_to_file, predict_proba, \
    write_evaluation_metrics_to_file, extract_case_ranking_performance_app, categorize_variable, categorize_age, \
    get_revision_id_of_original_case

def main(dir_output):
    if not exists(dir_output):
        makedirs(dir_output)
        logger.info(f'Created directory: {dir_output}')

    # load meta data containing aimedic id and label (whether the case was revised)
    meta_data_train = wr.s3.read_csv(FILENAME_TRAIN_SPLIT)
    meta_data_test = wr.s3.read_csv(FILENAME_TEST_SPLIT)

    # read in all data from DB and merge it with the labels from the meta data
    data_train = load_data(meta_data_train, load_diagnoses=True, load_procedures=True, only_revised_cases=False)
    data_test = load_data(meta_data_test, load_diagnoses=True, load_procedures=True, only_revised_cases=False)

    ## train current model
    # age bins-
    # encoded gender
    # number_of_sdx, number_of_chops, number_of_used_diags, number_of_used_chops, number_of_ccl_triggering_diags, number_of_grouper_relevant_diags
    # pccl
    # (effective) cw
    # mdc
    # duration of stay

    X_train_age_bins, age_labels = categorize_age(data_train['age_years'].values, data_train['age_days'].values)
    X_test_age_bins, _ = categorize_age(data_test['age_years'], data_test['age_days'].values)

    # X_train_age = data_train['age_years'].values.reshape((-1, 1))
    # X_test_age = data_test['age_years'].values.reshape((-1, 1))

    X_train_gender, gender_label, encoder_gender = categorize_variable(data_train, 'gender')
    X_test_gender, _, _ = categorize_variable(data_test, 'gender', encoder=encoder_gender)

    X_train_dos = data_train['duration_of_stay'].values.reshape((-1, 1))
    X_test_dos = data_test['duration_of_stay'].values.reshape((-1, 1))

    X_train_number_of_sdx, X_train_number_of_chops, X_train_number_of_used_diags, X_train_number_of_used_chops, X_train_number_of_ccl_triggering_diags, number_of_predictor_labels, pccl_train_original_case, effektive_cost_weight_train_original_case = extract_number_of(
        data_train)
    X_test_number_of_sdx, X_test_number_of_chops, X_test_number_of_used_diags, X_test_number_of_used_chops, X_test_number_of_ccl_triggering_diags, _, pccl_test_original_case, effektive_cost_weight_test_original_case = extract_number_of(
        data_test)

    data_train['effective_cost_weight_original'] = effektive_cost_weight_train_original_case
    data_test['effective_cost_weight_original'] = effektive_cost_weight_test_original_case
    X_train_cw = data_train['effective_cost_weight_original'].values.reshape((-1, 1))
    X_test_cw = data_test['effective_cost_weight_original'].values.reshape((-1, 1))

    data_train['pccl_original'] = pccl_train_original_case
    data_test['pccl_original'] = pccl_test_original_case
    X_train_pccl, pccl_label, encoder_pccl = categorize_variable(data_train, 'pccl_original')
    X_test_pccl, _, _ = categorize_variable(data_test, 'pccl_original', encoder=encoder_pccl)

    # # add discharge year
    # X_train_discharge_year, X_test_discharge_year, label_discharge_year = get_categorized_predictors(data_train, data_test, column='discharge_year')

    # # get nems points
    # X_train_nems, X_test_nems = get_continous_int_variable(data_train, data_test, 'nems_total')

    # # get IMC effort points
    # X_train_imc_effort_points, X_test_imc_effort_points = get_continous_int_variable(data_train, data_test, 'imc_effort_points')

    # # get IMC effort points
    # X_train_ventilation_hours, X_test_ventilation_hours = get_continous_int_variable(data_train, data_test, 'ventilation_hours')

    # define model input
    predictor_labels = list(np.concatenate([
        age_labels,
        # ['age'],
        gender_label,
        pccl_label,
        ['effective_cost_weight'],
        ['duration_of_stay'],
        number_of_predictor_labels,  # this contains all cases
        # label_discharge_year
        # ['nems']
        # ['IMC_effort_points']
        # ['ventilation_hours']
    ]))

    y_label = 'y_label_is_revised_case'
    X_train = np.hstack([
        X_train_age_bins,
        # X_train_age,
        X_train_gender,
        X_train_pccl,
        X_train_cw,
        X_train_dos,
        X_train_number_of_sdx,
        X_train_number_of_chops,
        X_train_number_of_used_diags,
        X_train_number_of_used_chops,
        X_train_number_of_ccl_triggering_diags,
        # X_train_discharge_year
        # X_train_nems
        # X_train_imc_effort_points
        # X_train_ventilation_hours
    ])
    y_train = data_train[y_label]

    X_test = np.hstack([
        X_test_age_bins,
        # X_test_age,
        X_test_gender,
        X_test_pccl,
        X_test_cw,
        X_test_dos,
        X_test_number_of_sdx,
        X_test_number_of_chops,
        X_test_number_of_used_diags,
        X_test_number_of_used_chops,
        X_test_number_of_ccl_triggering_diags,
        # X_test_discharge_year
        # X_test_nems
        # X_test_imc_effort_points
        # X_test_ventilation_hours
    ])
    y_test = data_test[y_label]

    # train the model
    model = train_rf_model(X_train, y_train)

    # evaluate model how well it predicts
    # write feature importances to file
    importances_rf = pd.Series(model.feature_importances_, index=X_train.columns)
    sorted_importances_rf = importances_rf.sort_values()
    sorted_importances_rf.to_csv(join(dir_output, 'rf_importances.csv'))

    logger.info("Write Random Forest Feature Importances to file")

    # predict on training and test data and get probability for the cases to be revisable
    prediction_probas_train = predict_proba(model, X_train)
    prediction_probas_test = predict_proba(model, X_test)

    # Plot feature importance of prediction
    # plt.figure()
    # importances_random_forest = pd.Series(train_random_forest.feature_importances_, index=X_test.columns)
    # sorted_importances_random_forest = importances_random_forest.sort_values()
    # sorted_importances_random_forest.plot(kind='barh', color='lightgreen')
    # save_figure_to_pdf_on_s3(plt, s3_bucket, os.path.join(dir_output, 'importances_random_forest.pdf'))

    # write f1, precision, recall to file
    write_evaluation_metrics_to_file(y_train, prediction_probas_train, y_test, prediction_probas_test,
                                     join(dir_output, 'metrics_random_forest.csv'), threshold=0.5)

    # write performance app output to file for case ranking comparisons
    extract_case_ranking_performance_app(data_test, prediction_probas_test,
                                         join(dir_output, 'performance_app_input_random_forest.csv'))

if __name__ == "__main__":
    main(dir_output=join(PROJECT_ROOT_DIR, 'results', 'results_current_model_ventilation_hours'))
