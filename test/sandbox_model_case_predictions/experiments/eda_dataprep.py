from os import makedirs
from os.path import join, exists

import numpy as np
import pandas as pd
from dataprep.eda import create_report
from loguru import logger

from src import ROOT_DIR
from test.sandbox_model_case_predictions.data_handler import load_data
from test.sandbox_model_case_predictions.experiments.random_forest import REVISED_CASE_IDS_FILENAME
from test.sandbox_model_case_predictions.utils import get_revised_case_ids, get_list_of_all_predictors, \
    prepare_train_eval_test_split

tag = '_without_OHE_and_flag_RAW'
dir_output = join(ROOT_DIR, 'results', 'eda_dataprep')
if not exists(dir_output):
    makedirs(dir_output)

DISCARDED_FEATURES = ('hospital', 'month_admission', 'month_discharge', 'year_discharge', 'mdc_OHE', 'hauptkostenstelle_OHE', 'is_emergency_case_RAW')
all_data = load_data(only_2_rows=True)
features_dir = join(ROOT_DIR, 'resources', 'features')
feature_filenames, encoders = get_list_of_all_predictors(all_data, features_dir, overwrite=False)
feature_names = sorted(list(feature_filenames.keys()))
feature_names = [feature_name for feature_name in feature_names
                 if not np.logical_or(np.logical_or(any(feature_name.startswith(discarded_feature) for discarded_feature in DISCARDED_FEATURES),feature_name.endswith('_OHE')), feature_name.endswith('_flag_RAW'))]

revised_cases_in_data = get_revised_case_ids(all_data, REVISED_CASE_IDS_FILENAME, overwrite=False)
reviewed_cases = revised_cases_in_data
y_is_revised = reviewed_cases['is_revised'].values
y_is_reviewed = reviewed_cases['is_reviewed'].values
sample_indices = reviewed_cases['index'].values

ind_train, ind_test, _, _, ind_hospital_leave_out, _ = \
    prepare_train_eval_test_split(dir_output=dir_output, revised_cases_in_data=revised_cases_in_data, only_reviewed_cases=True)

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

# define data frame containing features to investigate
df_data = pd.DataFrame(features, columns=feature_ids)
df_data['is_revised'] = y_is_revised
df_data['is_reviewed'] = y_is_reviewed

ind_train_test = np.concatenate([ind_train, ind_test])
create_report(df_data.iloc[ind_train_test]).save(join(dir_output, f"my_report{tag}"))