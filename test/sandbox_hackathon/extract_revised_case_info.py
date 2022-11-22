from os import makedirs
from os.path import join, exists

import awswrangler as wr
import pandas as pd

from src import PROJECT_ROOT_DIR
from src.service.bfs_cases_db_service import get_hospital_df
from src.service.database import Database
from test.sandbox_hackathon.constants import FILENAME_TRAIN_SPLIT, FILENAME_TEST_SPLIT
from test.sandbox_hackathon.utils import load_data


def main(dir_output):
    if not exists(dir_output):
        makedirs(dir_output)

    meta_data_train = wr.s3.read_csv(FILENAME_TRAIN_SPLIT)
    meta_data_test = wr.s3.read_csv(FILENAME_TEST_SPLIT)

    data = load_data(pd.concat([
        meta_data_train,
        meta_data_test
    ]), load_diagnoses=False, load_procedures=False, only_revised_cases=True)
    data = data.astype({'aimedic_id': 'string', 'hospital_id': 'string', 'clinic_id': 'string', 'patient_id': 'string', 'case_id': 'string', 'discharge_year': 'string'})

    with Database() as db:
        hospitals = get_hospital_df(db.session)
        hospitals = hospitals.astype({'hospital_id': 'string', 'hospital_name': 'string'})

    data = pd.merge(data, hospitals, on='hospital_id', how='outer')
    data_to_file = data[['aimedic_id', 'hospital_id', 'clinic_id', 'patient_id', 'case_id', 'discharge_year', 'hospital_name']].dropna()
    data_to_file.to_csv(join(dir_output, 'revised_case_info.csv'), index=False)


if __name__ == "__main__":
    main(dir_output=join(PROJECT_ROOT_DIR, 'results', 'revised_case_info'))
