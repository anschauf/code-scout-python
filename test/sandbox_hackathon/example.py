from os import makedirs
from os.path import join, exists

import awswrangler as wr

from src import PROJECT_ROOT_DIR
from test.sandbox_hackathon.constants import FILENAME_TRAIN_SPLIT, FILENAME_TEST_SPLIT, RANDOM_SEED
from test.sandbox_hackathon.utils import load_data, train_lr_model, write_model_coefs_to_file, predict_proba, \
    write_evaluation_metrics_to_file, extract_case_ranking_performance_app

dir_output = join(PROJECT_ROOT_DIR, 'results', 'test_results_dos_age')
if not exists(dir_output):
    makedirs(dir_output)

# load meta data containing aimedic id and label (whether the case was revised)
meta_data_train = wr.s3.read_csv(FILENAME_TRAIN_SPLIT)
meta_data_test = wr.s3.read_csv(FILENAME_TEST_SPLIT)

# read in all data from DB and merge it with the labels from the meta data
data_train = load_data(meta_data_train)
data_test = load_data(meta_data_test)

# do your feature engineering magic here
# for this example, duration of stay and age in years does the trick
predictor_labels = ['duration_of_stay', 'age_years']
y_label = 'y_label_is_revised_case'
X_train = data_train[predictor_labels].values
y_train = data_train[y_label].values
X_test = data_test[predictor_labels].values
y_test = data_test[y_label].values

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
