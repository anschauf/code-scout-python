import os
import sys
from loguru import logger
import awswrangler as wr
from src.revised_case_normalization.notebook_functions.global_configs import *
from src.revised_case_normalization.notebook_functions.normalize import normalize
from src.revised_case_normalization.notebook_functions.revise import revise
from src.revised_case_normalization.notebook_functions.group import group
from src.revised_case_normalization.notebook_functions.update_db import update_db


# local work directory
dir_cwd = os.getcwd()
# create a results folder
dir_output = os.path.join(dir_cwd, 'revised_case_DB_results')
if not os.path.isdir(dir_output):
        os.mkdir(dir_output)


file_info = FileInfo(os.path.join(dir_cwd, 'raw_data/HI_Aarau_Birshof_ST. Anna.xlsx'),
                     'Hirslanden Klinik Aarau', '2017', 'Aarau_2017')

print(file_info)

# Create a output file names
hospital_name = '_'.join(file_info.hospital_name_db.lower().split())
year = file_info.year
logger_file = f'{hospital_name}_{year}.log'
logger_file_path = os.path.join(dir_output, f'{hospital_name}_{year}.log')

logger = logger.add(logger_file_path)

with open(logger_file_path, 'w') as f:
    revised_cases_df = normalize(file_info)
    cols_to_join = list(VALIDATION_COLS)
    cols_to_join.remove(CASE_ID_COL)
    cols_to_join.append(NORM_CASE_ID_COL)
    # Ignore the patient ID in this dataset, because it is encrypted
    cols_to_join.remove(PATIENT_ID_COL)
    revised_cases, unmatched = revise(file_info, revised_cases_df,  validation_cols=cols_to_join)

revised_cases_df

cols_to_join = list(VALIDATION_COLS)
cols_to_join.remove(CASE_ID_COL)
cols_to_join.append(NORM_CASE_ID_COL)
# Ignore the patient ID in this dataset, because it is encrypted
cols_to_join.remove(PATIENT_ID_COL)

revised_cases.dtypes

if unmatched.shape[0] > 0:
    unmatched



revisions_update, diagnoses_update, procedures_update = group(revised_cases)
revisions_update[REVISION_DATE_COL] = '2022-12-31'



#update_db(revisions_update, diagnoses_update, procedures_update)

