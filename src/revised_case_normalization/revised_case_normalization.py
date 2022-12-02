import os
import sys

import pandas as pd
from loguru import logger
import awswrangler as wr
from src.revised_case_normalization.notebook_functions.global_configs import *
from src.revised_case_normalization.notebook_functions.revised_case_files_info import REVISED_CASE_FILES
from src.revised_case_normalization.notebook_functions.normalize import normalize
from src.revised_case_normalization.notebook_functions.revise import revise
from src.revised_case_normalization.notebook_functions.group import group
from src.revised_case_normalization.notebook_functions.update_db import update_db

# TODO refactoring the code to read data and save result into s3

# local work directory
dir_cwd = os.getcwd()
# create a results folder
dir_output = os.path.join(dir_cwd, 'revised_case_DB_results')
if not os.path.isdir(dir_output):
        os.mkdir(dir_output)



all_revision_list = list()
all_diagnoses_list = list()
all_procedure_list = list()


for file_info in REVISED_CASE_FILES:
    # Create an output file name for each based on hospital and year
    hospital_name = '_'.join(file_info.hospital_name_db.lower().split())
    year = file_info.year
    logger_file = f'{hospital_name}_{year}.log'
    logger_file_path = os.path.join(dir_output, f'{hospital_name}_{year}.log')
    logger.remove() # prevent to add logger message to previous files
    logger.add(logger_file_path, mode='w')

    print(f'{hospital_name} {year}')
    # handling the exception of case_id with fall number
    if hospital_name == 'hirslanden_klinik_zurich' and year == '2019':
        columns_to_rename = dict(COLUMNS_TO_RENAME)
        columns_to_rename.pop("admno")
        columns_to_rename['fall nummer'] = CASE_ID_COL
        revised_cases_df = normalize(file_info, columns_mapper=columns_to_rename)
    else:
        revised_cases_df = normalize(file_info)

    cols_to_join = list(VALIDATION_COLS)
    cols_to_join.remove(CASE_ID_COL)
    cols_to_join.append(NORM_CASE_ID_COL)
    # The patient ID is ignored, because it can be encrypted
    cols_to_join.remove(PATIENT_ID_COL)
    revised_cases, unmatched = revise(file_info, revised_cases_df, validation_cols=cols_to_join)
    logger.info(f'TYPES:\n{revised_cases.dtypes}')
    if unmatched.shape[0] > 0:
        unmatched
    num_row = revised_cases.shape[0]
    # special handling for a few cases which can not be grouped
    if hospital_name == 'hirslanden_klinik_zurich' and year == '2018':
        # The row 96 can not be grouped, we deleted it at the moment
        revised_cases.drop(96, inplace=True)
    if hospital_name == 'hirslanden_klinik_zurich' and year == '2016':
        # The row 43 can not be grouped, we deleted it at the moment
        revised_cases.drop(43, inplace=True)




    if num_row != revised_cases.shape[0]:
        num_row_deleted = num_row - revised_cases.shape[0]
        logger.info(f'{num_row_deleted} can not be grouped')

    revisions_update, diagnoses_update, procedures_update = group(revised_cases)
    revisions_update[REVISION_DATE_COL] = '2022-12-31'
    all_revision_list.append(revisions_update)
    all_diagnoses_list.append(diagnoses_update)
    all_procedure_list.append(procedures_update)


# concatenate all dataframe to one
all_revision_df = pd.concat(all_revision_list)
all_diagnoses_df = pd.concat(all_diagnoses_list)
all_procedure_df = pd.concat(all_procedure_list)

all_procedure_df.to_csv('all_revision_df.csv')
#update_db(revisions_update, diagnoses_update, procedures_update)

print(len(all_revision_df))