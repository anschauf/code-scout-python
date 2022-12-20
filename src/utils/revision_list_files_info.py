import pandas as pd
from src import ROOT_DIR
import os
from dataclasses import dataclass

@dataclass
class FolderHospitalYear:
    path: str
    hospital_name_db: str
    year: int


DIR_REVISION_LISTS = 's3://aimedic-patient-data/revision_lists' # s3 path
# DIR_REVISION_LISTS = os.path.join(ROOT_DIR, 'src/revision_lists')
CASE_IDS_CANDIDATES = ['admno', 'fall-id']


REVISION_LIST_INFO = [
    # FolderHospitalYear(os.path.join(DIR_REVISION_LISTS, 'Bern_Linde_2017'),
    #                  'Hirslanden Klinik Linde', 2017),
    #
    # FolderHospitalYear(os.path.join(DIR_REVISION_LISTS, 'Bern_Linde_2018'),
    #                  'Hirslanden Klinik Linde', 2018),
    #
    # FolderHospitalYear(os.path.join(DIR_REVISION_LISTS, 'Aarau_2018'),
    #                  'Hirslanden Klinik Aarau', 2018),
    #
    # FolderHospitalYear(os.path.join(DIR_REVISION_LISTS, 'Beau_Site_2017'),
    #                    'Hirslanden Klinik Beau-Site', 2017),
    #
    # FolderHospitalYear(os.path.join(DIR_REVISION_LISTS, 'Beau_Site_2018'),
    #                    'Hirslanden Klinik Beau-Site', 2018),
    # # no data find in bfs 2018
    # FolderHospitalYear(os.path.join(DIR_REVISION_LISTS, 'KSW_2018'),
    #                    'Kantonsspital Winterthur', 2018),
    #
    # try to match 2019
    FolderHospitalYear(os.path.join(DIR_REVISION_LISTS, 'KSW_2018'),
                       'Kantonsspital Winterthur', 2019),

    # FolderHospitalYear(os.path.join(DIR_REVISION_LISTS, 'Prüflisten_KSSG_2021'),
    #                    'KSSG St. Gallen', 2021),
    # No data in this folder
    # FolderHospitalYear(os.path.join(DIR_REVISION_LISTS, 'Prüflisten_SR2_2019'),
    #                    '', 2019),

    # FolderHospitalYear(os.path.join(DIR_REVISION_LISTS, 'Prüflisten_SR3_2019'),
    #                    'KSSG Spital Linth', 2019),
    #
    # FolderHospitalYear(os.path.join(DIR_REVISION_LISTS, 'Prüflisten_SR4_2019'),
    #                    'KSSG Spitalregion Fürstenland Toggenburg', 2019),
    #
    # FolderHospitalYear(os.path.join(DIR_REVISION_LISTS, 'Salem_2017'),
    #                    'Hirslanden Salem-Spital', 2017),
    #
    # FolderHospitalYear(os.path.join(DIR_REVISION_LISTS, 'Salem_2018'),
    #                    'Hirslanden Salem-Spital', 2018),
    #
    # FolderHospitalYear(os.path.join(DIR_REVISION_LISTS, 'USZ_2018'),
    #                    'Universitätsspital Zürich', 2018),
]
