import os
import time

import awswrangler as wr
import boto3
from loguru import logger

from src import ROOT_DIR
from src.service.bfs_cases_db_service import get_sociodemographics_for_hospital_year, get_all_revised_cases, \
    update_reviewed_for_sociodemographic_ids
from src.service.database import Database
from src.utils.general_utils import __remove_prefix_and_bucket_if_exists
from src.utils.normalize import remove_leading_zeros
from src.utils.read_csv_excel import read_csv_excel
from src.utils.revision_list_files_info import REVISION_LIST_INFO, FolderHospitalYear, CASE_IDS_CANDIDATES
import pandas as pd


def revision_list_matching(*,
                           dir_files_to_import: list[FolderHospitalYear],
                           s3_bucket: str = 'aimedic-patient-data',
                           s3_bucket_logs: str = 'code_scout'
                           ):
    # Create a new local log file at the root of the project
    log_path = os.path.join(ROOT_DIR, 'logs')
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    time_str = time.strftime("%Y%m%d-%H%M")
    log_filename = f'{log_path}/revised_cases_import_{time_str}.log'
    logger.add(log_filename, mode='w')

    # Connect to s3
    s3 = boto3.resource('s3')
    s3_bucket = s3.Bucket(s3_bucket)

    for idx, hospital_year in enumerate(dir_files_to_import):
        n_files = len(dir_files_to_import)
        path_folder = hospital_year.path
        hospital = hospital_year.hospital_name_db
        year = hospital_year.year

        logger.info(f"#{idx + 1}/{n_files}: Working on '{hospital}' ({year}) ...")

        with Database() as db:
            # get sociodemographic_id for revised cases
            all_revised_case = get_all_revised_cases(db.session)
            sociodemographic_id_revised_cases = all_revised_case['sociodemographic_id']
            # get the sociodemographic table for hospital year
            sociodemographics_df = get_sociodemographics_for_hospital_year(hospital, year, db.session)
        # create a case_id_norm column without leading 0
        sociodemographics_df['case_id_norm'] = sociodemographics_df['case_id'].apply(remove_leading_zeros)
        #  extract sociodemographic_id and so case_id_norm
        revised_sociodemographics_id_case_id = sociodemographics_df[["sociodemographic_id", "case_id_norm"]]

        revision_case_ids_norm_all = list()

        # get files from s3 using folder name as prefix
        prefix = __remove_prefix_and_bucket_if_exists(path_folder)
        logger.info(f'Work on the files in the folder: {prefix}')
        for file_obj in s3_bucket.objects.filter(Prefix=prefix):
            obj_key = file_obj.key
            if obj_key.endswith('/'):
                continue

            file_name = obj_key.split('/')[-1]  # get the file name of excel or csv
            logger.info(f'Working on the file {file_name}')
            df = read_csv_excel(path_folder, file_name)

            df.columns = [col.lower() for col in df.columns]

            # Find the case_id column
            case_id_col = set(CASE_IDS_CANDIDATES).intersection(df.columns)

            if len(case_id_col) == 1:
                case_id = case_id_col.pop()
            else:
                print(df.columns)
                logger.warning("No case_id or multiple case_id columns found, check the excel")
            # delete rows with case_id as nan
            df.dropna(subset=case_id, inplace=True)
            revision_case_ids = df[case_id].tolist()

            revision_case_ids_norm = [id.strip("'").strip("0") for id in revision_case_ids]
            logger.info(f"There are {len(revision_case_ids_norm)} number of cases in the file {file_name}")
            revision_case_ids_norm_all.extend(revision_case_ids_norm)
        # Get only one copy of case_id of each hospital year
        revision_case_ids_norm_all_set = set(revision_case_ids_norm_all)

        revision_case_id_df = pd.DataFrame(revision_case_ids_norm_all_set, columns=['case_id_norm'])

        df_reviewed = pd.merge(revision_case_id_df, revised_sociodemographics_id_case_id,
                               how='inner', on='case_id_norm')

        logger.info(f'Matched {len(df_reviewed)} number of reviewed cases to database')

        # Check if the sociodemagraphic_ids in revised case, not, update the reviewed as true in the revision table
        sociodemographic_id_reviewed_cases = df_reviewed['sociodemographic_id'].tolist()
        # Get the sociodemographic_ids not in revised cases
        sociodemographic_id_reviewed_no_revised = set(sociodemographic_id_reviewed_cases).difference(
            set(sociodemographic_id_revised_cases))

        logger.info(
            f'There are {len(sociodemographic_id_reviewed_no_revised)} number of cases for {hospital} in {year} needs to updated as reviewed in the DB')

        # # updating reviewed as true in revision table in db
        # with Database() as db:
        #     update_reviewed_for_sociodemographic_ids(sociodemographic_id_reviewed_no_revised, db.session)

        # upload log file to s3
        # s3 = boto3.resource('s3')
        # log_path_s3 = 'revision_list_logs'
        # log_filename_s3 = log_filename.replace(log_path, log_path_s3)
        # filename = __remove_prefix_and_bucket_if_exists(log_filename_s3)
        # s3_object = s3.Object(s3_bucket_logs, filename)
        # s3_object.put(Body=open(log_filename, 'rb'))


if __name__ == '__main__':
    revision_list_matching(dir_files_to_import=REVISION_LIST_INFO)
