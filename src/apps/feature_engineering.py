from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from data_model.feature_engineering import HAS_VENTILATION_HOURS_COL
from revised_case_normalization.notebook_functions.global_configs import VENTILATION_HOURS_COL
from src.apps.feature_engineering.ccl_sensitivity import calculate_delta_pccl
from src.apps.feature_engineering.utils import create_feature_engineering_table, store_features_in_db, validate_app_args
from src.service.bfs_cases_db_service import read_cases_in_chunks
from src.service.database import Database


def create_all_features(*, chunksize: int, n_rows: Optional[int] = None):
    chunksize, n_rows = validate_app_args(chunksize, n_rows)
    columns_to_select = create_feature_engineering_table()

    all_features = None

    with Database() as db:
        for cases in read_cases_in_chunks(db.session, n_rows=n_rows, chunksize=chunksize):
            # --- Engineer features ---
            cases = calculate_delta_pccl(cases, delta_value_for_max=np.nan)
            cases[HAS_VENTILATION_HOURS_COL] = cases[VENTILATION_HOURS_COL] > 0



            # --- Store the data in memory ---
            if all_features is None:
                all_features = cases.copy()
            else:
                all_features = pd.concat((all_features, cases), axis='index', copy=False)

    # --- Store the data in the DB ---
    with Database() as db:
        store_features_in_db(all_features, columns_to_select, db.session)

    logger.success('completed')


if __name__ == '__main__':
    create_all_features(
        chunksize=5,
        n_rows=10
    )
