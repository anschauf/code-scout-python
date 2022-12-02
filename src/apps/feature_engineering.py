import numpy as np
import pandas as pd
from beartype import beartype
from loguru import logger

from src.data_model.feature_engineering import FeatureEngineering, DELTA_CCL_TO_NEXT_PCCL_COL
from src.revised_case_normalization.notebook_functions.global_configs import REVISION_ID_COL
from src.service.bfs_cases_db_service import read_cases, create_table
from src.service.database import Database


def create_all_features():
    cases = read_cases(n_rows=100)

    # --- Engineer features ---
    cases = calculate_delta_pccl(cases, delta_value_for_max=np.nan)

    # --- Store the data ---
    store_features_in_db(cases)
    logger.success('completed')


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
    table_name = f'{FeatureEngineering.__table__.schema}.{FeatureEngineering.__tablename__}'
    logger.info(f"Storing {data.shape[0]} rows to '{table_name}' ...")

    # List the columns in the DB table
    columns = list(FeatureEngineering.__table__.columns)
    column_names = list()
    for column in columns:
        if not column.primary_key:
            column_names.append(column.name)

    # Select columns in the DataFrame which also appear in the DB table
    data_to_store = (data[column_names]
                     .sort_values(REVISION_ID_COL, ascending=True)
                     .to_dict(orient='records'))

    logger.info(f'Selected the columns: {column_names}')

    with Database() as db:
        # noinspection PyTypeChecker
        create_table(FeatureEngineering, db.session, overwrite=True)

        insert_statement = FeatureEngineering.__table__.insert().values(data_to_store)
        db.session.execute(insert_statement)
        db.session.commit()

    logger.success(f"Stored {len(data_to_store)} rows to '{table_name}'")


if __name__ == '__main__':
    create_all_features()
