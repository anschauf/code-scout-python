import os

from loguru import logger
from pandas import DataFrame

from src.service.bfs_cases_db_service import get_sociodemographics_for_hospital_year, get_all_revised_cases
from src.service.database import Database
from src.utils.revision_list_files_info import REVISION_LIST_INFO, DIR_REVISION_LISTS
import pandas as pd

case_ids_candidates = ['admno', 'fall-id']

with Database() as db:
   all_revised_case = get_all_revised_cases(db.session)
   sociodemographic_id_revised_cases = all_revised_case['sociodemographic_id']



def remove_leading_zeros(s: str) -> str:
    return s.lstrip('0')


for i, hospital_year in enumerate(REVISION_LIST_INFO):
    path_folder = hospital_year.path
    hospital = hospital_year.hospital_name_db
    year = hospital_year.year
    case_ids_norm_all = list()
    with Database() as db:
        sociodemographics_df = get_sociodemographics_for_hospital_year(hospital, year, db.session)
    sociodemographics_df['case_id_norm'] = sociodemographics_df['case_id'].apply(remove_leading_zeros)
    sociodemographics_id_case_id = sociodemographics_df[["sociodemographic_id", "case_id_norm"]]
    for file in os.listdir(path_folder):
        # process csv files
        if file.endswith('.csv'):
            file_path = os.path.join(path_folder, file)
            df = pd.read_csv(file_path, dtype='string[pyarrow]', sep=";", encoding='ISO-8859-1')
            df.columns = [col.lower() for col in df.columns]
            # get case_id from fall-id
            case_id_col = set()
            try:
                case_id_col = set(case_ids_candidates).intersection(df.columns)
                if len(case_id_col) == 1:
                    case_id = case_id_col.pop()
                if len(case_id_col) == 0:
                    print('no case_id column is found in the file')
                    print(df.columns)
                if len(case_id_col) > 1:
                    print(df.columns)
                    print('more than two case_id columns, something might be wrong')
            except:
                print(df.columns)

            case_ids = df[case_id].tolist()

            case_ids_norm = [id.strip("'").strip("0") for id in case_ids]
            case_ids_norm_all.extend(case_ids_norm)
            case_ids_norm_all_set = set(case_ids_norm_all)

        # if file.endswith('.xlsx'):





        case_id_df = pd.DataFrame(case_ids_norm_all_set, columns=['case_id'])


        df_reviewed = pd.merge(case_id_df, sociodemographics_id_case_id,
                          how='inner', left_on='case_id', right_on='case_id_norm')

        logger.info(f'Matched {len(df_reviewed)} number of reviewed cases to database')

        # Check if the sociodemagraphic_ids in revised case, not, update the reviewed as true in the revision table
        sociodemographic_id_reviewed_cases = df_reviewed['sociodemographic_id'].tolist()
        sociodemographic_id_reviewed_no_revised = set(sociodemographic_id_reviewed_cases).difference(set(sociodemographic_id_revised_cases))

        logger.info(f'There are {len(sociodemographic_id_reviewed_no_revised)} number of cases for {hospital} in {year} needs to updated as reviewed in the DB')





print('')
