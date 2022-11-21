from os import makedirs
from os.path import join, exists

import awswrangler as wr
import numpy as np

from src import PROJECT_ROOT_DIR
from test.sandbox_hackathon.constants import FILENAME_TRAIN_SPLIT, FILENAME_TEST_SPLIT, RANDOM_SEED
from test.sandbox_hackathon.utils import load_data, train_lr_model, write_model_coefs_to_file, predict_proba, \
    write_evaluation_metrics_to_file, extract_case_ranking_performance_app, categorize_variable, categorize_age


def main(dir_output):
    if not exists(dir_output):
        makedirs(dir_output)

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

    X_train_gender, gender_label, encoder_gender = categorize_variable(data_train, 'gender')
    X_test_gender, _, _ = categorize_variable(data_test, 'gender', encoder=encoder_gender)

    data_train['pccl_revised'] = data_train['pccl'].apply(lambda x: x[0] if (len(x) == 1) else x[-1])
    data_test['pccl_revised'] = data_test['pccl'].apply(lambda x: x[0] if (len(x) == 1) else x[-1])
    X_train_pccl, pccl_label, encoder_pccl = categorize_variable(data_train, 'pccl_revised')
    X_test_pccl, _, _ = categorize_variable(data_test, 'pccl_revised', encoder=encoder_pccl)

    data_train['effective_cost_weight_revised'] = data_train['effective_cost_weight'].apply(lambda x: x[0] if (len(x) == 1) else x[-1])
    data_test['effective_cost_weight_revised'] = data_test['effective_cost_weight'].apply(lambda x: x[0] if (len(x) == 1) else x[-1])
    X_train_cw = data_train['effective_cost_weight_revised'].values.reshape((-1,1))
    X_test_cw = data_test['effective_cost_weight_revised'].values.reshape((-1,1))

    X_train_dos = data_train['duration_of_stay'].values.reshape((-1,1))
    X_test_dos = data_test['duration_of_stay'].values.reshape((-1,1))

    # create number of code fields
    X_train_number_of_sdx = data_train['code_diagnoses'].apply(lambda x: len(x) - 1 if isinstance(x, list) else 0).values.reshape((-1,1))
    X_test_number_of_sdx = data_test['code_diagnoses'].apply(lambda x: len(x) - 1 if isinstance(x, list) else 0).values.reshape((-1,1))

    X_train_number_of_chops = data_train['code_procedures'].apply(lambda x: len(x) if isinstance(x, list) else 0).values.reshape((-1,1))
    X_test_number_of_chops = data_test['code_procedures'].apply(lambda x: len(x) if isinstance(x, list) else 0).values.reshape((-1,1))

    X_train_number_of_used_diags = data_train['is_grouper_relevant_diagnoses'].apply(lambda x: sum(x) if isinstance(x, list) else 0).values.reshape((-1,1))
    X_test_number_of_used_diags = data_test['is_grouper_relevant_diagnoses'].apply(lambda x: sum(x) if isinstance(x, list) else 0).values.reshape((-1,1))

    X_train_number_of_used_chops = data_train['is_grouper_relevant_procedures'].apply(lambda x: sum(x) if isinstance(x, list) else 0).values.reshape((-1,1))
    X_test_number_of_used_chops = data_test['is_grouper_relevant_procedures'].apply(lambda x: sum(x) if isinstance(x, list) else 0).values.reshape((-1,1))

    X_train_number_of_ccl_triggering_diags = data_train['ccl_diagnoses'].apply(lambda x: sum(np.asarray(x)>0) if isinstance(x, list) else 0).values.reshape((-1,1))
    X_test_number_of_ccl_triggering_diags = data_test['ccl_diagnoses'].apply(lambda x: sum(np.asarray(x)>0) if isinstance(x, list) else 0).values.reshape((-1,1))

    # define model input
    predictor_labels = list(np.concatenate([age_labels, gender_label, pccl_label, ['effective_cost_weight', 'duration_of_stay', 'number_of_sdx', 'number_of_chops', 'number_of_used_diags', 'number_of_used_chops', 'number_of_ccl_triggering_diags']]))
    y_label = 'y_label_is_revised_case'
    X_train = np.hstack([X_train_age_bins, X_train_gender, X_train_pccl, X_train_cw, X_train_dos, X_train_number_of_sdx, X_train_number_of_chops, X_train_number_of_used_diags, X_train_number_of_used_chops, X_train_number_of_ccl_triggering_diags])
    y_train = data_train[y_label]
    X_test = np.hstack([X_test_age_bins, X_test_gender, X_test_pccl, X_test_cw, X_test_dos, X_test_number_of_sdx, X_test_number_of_chops, X_test_number_of_used_diags, X_test_number_of_used_chops, X_test_number_of_ccl_triggering_diags])
    y_test = data_test[y_label]

    # train the model
    model = train_lr_model(X_train, y_train, penalty='l1', class_weight='balanced', solver='liblinear', random_state=RANDOM_SEED, fit_intercept=False)

    # evaluate model how well it predicts
    # write model coefficients to file
    write_model_coefs_to_file(model, join(dir_output, 'model_coefs.csv'), predictor_labels=predictor_labels)

    # predict on training and test data and get probability for the cases to be revisable
    prediction_probas_train = predict_proba(model, X_train)
    prediction_probas_test = predict_proba(model, X_test)

    # write f1, precision, recall to file
    write_evaluation_metrics_to_file(y_train, prediction_probas_train, y_test, prediction_probas_test, join(dir_output, 'metrics.csv'), threshold=0.5)

    # write performance app output to file for case ranking comparisons
    extract_case_ranking_performance_app(data_test, prediction_probas_test, join(dir_output, 'performance_app_input.csv'))

if __name__ == "__main__":
    main(dir_output=join(PROJECT_ROOT_DIR, 'results', 'results_current_model'))