from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from apps.feature_engineering.engineer_features import engineer_features
from apps.feature_engineering.medications import get_atc_codes
from revised_case_normalization.notebook_functions.global_configs import VENTILATION_HOURS_COL
from src.apps.feature_engineering.ccl_sensitivity import calculate_delta_pccl
from src.apps.feature_engineering.utils import create_feature_engineering_table, store_features_in_db, validate_app_args
from src.data_model.feature_engineering import AGE_BINNED_COL, EMERGENCY_COL, HAS_HOURS_IN_ICU_COL, \
    HAS_IMC_EFFORT_POINTS_COL, HAS_NEMS_POINTS_COL, HAS_VENTILATION_HOURS_COL, VENTILATION_HOURS_ADRG_NO_A_COL
from src.service.bfs_cases_db_service import read_cases_in_chunks,\
    read_cases_by_sociodemographic_id
from src.service.database import Database


def create_all_features(*, chunksize: int, n_rows: Optional[int] = None):
    chunksize, n_rows = validate_app_args(chunksize, n_rows)
    columns_to_select = create_feature_engineering_table()

    all_features = None

    with Database() as db:
        for cases in read_cases_by_sociodemographic_id(db.session, n_rows=n_rows, chunksize=chunksize, lower=800000, upper=850000):

            # Calculating delta pccl
            cases = calculate_delta_pccl(cases, delta_value_for_max=np.nan)

            # Engineer Features
            cases = engineer_features(cases)

            # --- Store the data in memory ---
            if all_features is None:
                all_features = cases.copy()[columns_to_select]
            else:
                all_features = pd.concat((all_features, cases[columns_to_select]), axis='index', copy=False)

    # --- Store the data in the DB ---
    with Database() as db:
        store_features_in_db(all_features, chunksize, db.session)

    logger.success('completed')

if __name__ == '__main__':
    create_all_features(
        chunksize=10,
        n_rows=100
    )
