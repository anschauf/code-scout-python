import os
import time

import awswrangler as wr
import boto3
import pandas as pd
from beartype import beartype
from loguru import logger

from src import ROOT_DIR
from src.models.sociodemographics import SOCIODEMOGRAPHIC_ID_COL
from src.service.bfs_cases_db_service import get_duration_of_stay_df
from src.service.database import Database
from src.utils.general_utils import __remove_prefix_and_bucket_if_exists
from src.utils.global_configs import *
from src.utils.group import group
from src.utils.normalize import normalize
from src.utils.revise import revise
from src.utils.revised_case_files_info import DIR_REVISED_CASES, FileInfo, FILES_FALL_NUMMER, FILES_FID, \
    REVISED_CASE_FILES


@beartype
def load_and_apply_revisions(*,
                             files_to_import: list[FileInfo],
                             s3_bucket: str = 'aimedic-patient-data'
                             ):
    # Create a new local log file at the root of the project
    log_path = os.path.join(ROOT_DIR, 'logs')
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    time_str = time.strftime("%Y%m%d-%H%M")
    log_filename = f'{log_path}/revised_cases_import_{time_str}.log'
    logger.add(log_filename, mode='w')

    # Initialize the lists to collect all the results from all the datasets
    all_revision_list = list()
    all_diagnoses_list = list()
    all_procedure_list = list()

    n_files = len(files_to_import)
    for idx, file_info in enumerate(files_to_import):
        # Create an output file name for each based on hospital and year
        hospital_name = '_'.join(file_info.hospital_name_db.lower().split())
        year = file_info.year

        logger.info(f"#{idx + 1}/{n_files}: Working on '{hospital_name}' ({year}) ...")
        # handling the sheet using fall number as case_id
        if {hospital_name: year} in FILES_FALL_NUMMER:
            columns_to_rename = dict(COLUMNS_TO_RENAME)
            columns_to_rename.pop("admno")
            columns_to_rename['fall nummer'] = CASE_ID_COL
            revised_cases_df = normalize(file_info, columns_mapper=columns_to_rename)

        # handling sheets using fid as case_id
        elif {hospital_name: year} in FILES_FID:
            columns_to_rename = dict(COLUMNS_TO_RENAME)
            columns_to_rename.pop("admno")
            columns_to_rename['fid'] = CASE_ID_COL
            revised_cases_df = normalize(file_info, columns_mapper=columns_to_rename)

            # Replace case_id with mapped case_ids from DtoD for kantonsspital winterthur 2018'
            if year == '2018':
                case_id_mapped = wr.s3.read_excel(
                    os.path.join(DIR_REVISED_CASES, 'case_id_mapping/case_id_mapping_KSW_2018.xlsx')).astype(str)
                revised_cases_df = pd.merge(revised_cases_df, case_id_mapped, on="case_id", how="left")
                revised_cases_df = revised_cases_df.drop('case_id_norm', axis=1)
                revised_cases_df.rename(columns={'case_id_mapped': 'case_id_norm'}, inplace=True)
        # sheets using admno as case_id
        else:
            revised_cases_df = normalize(file_info)

        cols_to_join = list(VALIDATION_COLS)
        cols_to_join.remove(CASE_ID_COL)
        cols_to_join.append(NORM_CASE_ID_COL)
        # The patient ID is ignored, because it can be encrypted
        cols_to_join.remove(PATIENT_ID_COL)

        try:
            revised_cases, _ = revise(file_info, revised_cases_df, validation_cols=cols_to_join)
        except ValueError:
            logger.warning(f'There is no data for the hospital {hospital_name} in {year}')
            continue

        # Convert datetime from datetime64[ns] to string type
        revision_date = revised_cases[REVISION_DATE_COL].astype(str).values.tolist()
        sociodemo_id = revised_cases[SOCIODEMOGRAPHIC_ID_COL].values.tolist()
        sociodemo_id_revision_date = dict(zip(sociodemo_id, revision_date))

        revisions_update, diagnoses_update, procedures_update = group(revised_cases)

        # get the original revision date after group the cases
        revisions_update[REVISION_DATE_COL] = revisions_update[SOCIODEMOGRAPHIC_ID_COL].astype(int).map(lambda x: sociodemo_id_revision_date.get(x))
        if revisions_update[REVISION_DATE_COL].isna().sum() > 0:
            raise ValueError(f'There is null values in revision date column for {hospital_name} {year}')

        all_revision_list.append(revisions_update)
        all_diagnoses_list.append(diagnoses_update)
        all_procedure_list.append(procedures_update)

    all_revision_df = pd.concat(all_revision_list)
    all_diagnoses_df = pd.concat(all_diagnoses_list)
    all_procedure_df = pd.concat(all_procedure_list)

    # Get the duration of stay ID from the "legacy code"
    with Database() as db:
        duration_of_stay_df = get_duration_of_stay_df(db.session)

    all_revision_df = pd.merge(
        # Drop the `dos_id` column because it comes from the `duration_of_stay_df` after the join
        all_revision_df.drop(columns='dos_id'),
        duration_of_stay_df.drop(columns='description'),  # Drop unused column
        how='left',
        left_on='duration_of_stay_legacy_code', right_on='dos_legacy_code')

    # Remove join keys
    all_revision_df.drop(columns=['duration_of_stay_legacy_code', 'dos_legacy_code'], inplace=True)

    # Set all the `reviewed` and `revised` flags to True
    all_revision_df['reviewed'] = True
    all_revision_df['revised'] = True

    num_revision = len(all_revision_df)
    logger.info(f'Number of revised cases: {num_revision}')

    unique_revisions_df = all_revision_df.drop_duplicates(subset=SOCIODEMOGRAPHIC_ID_COL)
    num_unique_revisions = unique_revisions_df.shape[0]
    if num_revision > num_unique_revisions:
        num_dup = num_revision - num_unique_revisions
        raise Exception(f'There are {num_dup} duplicates / {num_unique_revisions} revised cases')

    # update_db(all_revision_df, all_diagnoses_df, all_procedure_df)
    logger.success('done')

    # upload log file to s3
    s3 = boto3.resource('s3')
    log_path_s3 = os.path.join(DIR_REVISED_CASES, 'logs')
    log_filename_s3 = log_filename.replace(log_path, log_path_s3)
    filename = __remove_prefix_and_bucket_if_exists(log_filename_s3)
    s3_object = s3.Object(s3_bucket, filename)
    s3_object.put(Body=open(log_filename, 'rb'))

    # delete local log_file
    # if os.path.exists(log_filename):
    #     os.remove(log_filename)
    # else:
    #     print("The logger file was not created properly")
    # # delete the local log folder after uploading log_file to s3
    # try:
    #     os.rmdir(log_path)
    # except OSError:
    #     raise Warning('Your local log folder can not be deleted because it was not empty')


if __name__ == '__main__':
    load_and_apply_revisions(files_to_import=REVISED_CASE_FILES)
