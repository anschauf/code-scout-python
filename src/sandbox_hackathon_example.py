from os import makedirs
from os.path import join, exists

import pandas as pd

from src.sandbox_hackathon_utils import load_data, train_lr_model, write_model_coefs_to_file, predict_proba, \
    write_evaluation_metrics_to_file, extract_case_ranking_performance_app

dir_output = 'test_results_dos_age'
if not exists(dir_output):
    makedirs(dir_output)

# load meta data containing aimedic id and label (whether the case was revised)
meta_data_train = pd.read_csv(join('aimedic_id_label_data_train.csv'))
meta_data_test = pd.read_csv(join('aimedic_id_label_data_test.csv'))

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
model = train_lr_model(X_train, y_train)

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


print('')
