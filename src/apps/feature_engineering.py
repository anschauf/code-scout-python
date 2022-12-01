import numpy as np
import pandas as pd
from beartype import beartype
from loguru import logger

from src.data_model.feature_engineering import FeatureEngineering, DELTA_CCL_TO_NEXT_PCCL_COL
from src.service.bfs_cases_db_service import read_cases, create_table
from src.service.database import Database


def create_all_features():
    cases = read_cases(n_rows=100)

    # --- Engineer features ---
    cases = calculate_delta_pccl(cases, delta_value_for_max=np.nan)

    # --- Store the data ---
    store_features_in_db(cases)
    logger.success('done')


@beartype
def calculate_delta_pccl(cases: pd.DataFrame, *, delta_value_for_max: float = 0.0) -> pd.DataFrame:
    def _calculate_delta_pccl(row):
        # The max PCCL from 2022 is 6 instead of 4
        if row['discharge_year'] <= 2021:
            max_pccl = 4
        else:
            max_pccl = 6

        current_pccl = row['pccl']
        if current_pccl == max_pccl:
            row[DELTA_CCL_TO_NEXT_PCCL_COL] = delta_value_for_max

        else:
            raw_pccl = row['raw_pccl']
            target_pccl = current_pccl + 1
            target_raw_pccl = target_pccl - 0.49
            row[DELTA_CCL_TO_NEXT_PCCL_COL] = target_raw_pccl - raw_pccl

        return row

    logger.info('Calculating the delta CCL to reach the next PCCL value ...')
    cases = cases.apply(_calculate_delta_pccl, axis=1)
    logger.success('done')
    return cases


def store_features_in_db(data: pd.DataFrame):
    with Database() as db:
        create_table(FeatureEngineering, db.session, overwrite=True)

        print('')




if __name__ == '__main__':
    create_all_features()
