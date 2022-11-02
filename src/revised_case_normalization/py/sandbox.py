import sys

import pandas as pd
import numpy as np

from loguru import logger

sys.path.insert(0, '/home/jovyan/work')

from src.revised_case_normalization.py.global_configs import *
from src.revised_case_normalization.py.normalize import normalize, remove_leading_zeros
from src.service import bfs_cases_db_service as bfs_db
from src.service.bfs_cases_db_service import session, get_sociodemographics_for_hospital_year, get_earliest_revisions_for_aimedic_ids, get_codes, apply_revisions
#from src.revised_case_normalization.py.format_for_grouper import format_for_grouper

file_info = FileInfo(
        os.path.join(ROOT_DIR, 'raw_data/HI_Aarau_Birshof_ST. Anna.xlsx'),
        'Hirslanden St. Anna',
        '2018',
        ['KOPIE_Änderungen_ST. Anna_2018'])

print(file_info)

df_revised_case_d2d = normalize(file_info, 0)

df_revised_case_d2d

print("")