import os
import sys
from os.path import join

from loguru import logger

from src import ROOT_DIR
from test.sandbox_model_case_predictions.data_handler import load_data
from test.sandbox_model_case_predictions.utils import get_list_of_all_predictors, get_revised_case_ids

OVERWRITE_REVISED_CASE_IDs = False
OVERWRITE_FEATURE_FILES = True


features_dir = join(ROOT_DIR, 'resources', 'features')
revised_case_ids_filename = join(ROOT_DIR, 'resources', 'data', 'revised_case_ids.csv')

should_load_data = OVERWRITE_REVISED_CASE_IDs or not os.path.exists(revised_case_ids_filename) or OVERWRITE_FEATURE_FILES
all_data = load_data(only_2_rows=not should_load_data)
revised_cases_in_data = get_revised_case_ids(all_data, revised_case_ids_filename, overwrite=OVERWRITE_REVISED_CASE_IDs)
feature_filenames, encoders = get_list_of_all_predictors(all_data, features_dir, overwrite=OVERWRITE_FEATURE_FILES)

feature_names = sorted(list(feature_filenames.keys()))
n_features = len(feature_names)
logger.success(f'Created {n_features} features')

sys.exit(0)
