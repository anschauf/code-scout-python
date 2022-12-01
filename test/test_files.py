import numpy as np
import pandas as pd

from src.data_model.feature_engineering import FeatureEngineering
from src.revised_case_normalization.notebook_functions.global_configs import GROUPER_FORMAT_COL, AIMEDIC_ID_COL, \
    NEW_PRIMARY_DIAGNOSIS_COL
from src.revised_case_normalization.notebook_functions.group import format_for_grouper
from src.service.aimedic_grouper import group_batch_group_cases
from src.service.bfs_cases_db_service import create_table, get_sociodemographics_for_hospital_year, \
    get_earliest_revisions_for_aimedic_ids, get_codes
from src.service.database import Database

with Database() as db:

    # create_table(FeatureEngineering, db.session, overwrite=True)
    df_socio = get_sociodemographics_for_hospital_year(hospital_name='Hirslanden Klinik Zurich', year=2019, session=db.session)[:100]

    original_revision_ids = get_earliest_revisions_for_aimedic_ids(df_socio[AIMEDIC_ID_COL].astype(int).values.tolist(), db.session)
    df_codes = get_codes(original_revision_ids, db.session)
    df_codes.rename(columns={'old_pd': NEW_PRIMARY_DIAGNOSIS_COL}, inplace=True)


df_cases = pd.merge(df_socio, df_codes, on=AIMEDIC_ID_COL, how='left')

formatted = format_for_grouper(df_cases)
grouper_format = formatted[GROUPER_FORMAT_COL].values.tolist()
dfs = group_batch_group_cases(grouper_format)
print('')
