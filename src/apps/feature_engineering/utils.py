from typing import Optional

import pandas as pd
from beartype import beartype
from loguru import logger
from sqlalchemy.orm import Session

from src.data_model.feature_engineering import FeatureEngineering
from src.data_model.sociodemographics import Sociodemographics
from src.revised_case_normalization.notebook_functions.global_configs import REVISION_ID_COL
from src.service.bfs_cases_db_service import create_table
from src.service.database import Database


FEATURE_ENGINEERING_TABLE_NAME = f'{FeatureEngineering.__table__.schema}.{FeatureEngineering.__tablename__}'


def validate_app_args(chunksize: int, n_rows: Optional[int]):
    if chunksize <= 0:
        raise ValueError(f'chunksize must be > 0')

    if n_rows is None:
        # Read it from the DB
        with Database() as db:
            n_rows = db.session.query(Sociodemographics).count()

    if n_rows <= 0:
        raise ValueError(f'n_rows must be > 0')

    logger.info(f'Analyzing {n_rows} rows in the DB, {chunksize} at a time ...')

    return chunksize, n_rows



def create_feature_engineering_table() -> list:
    with Database() as db:
        # noinspection PyTypeChecker
        create_table(FeatureEngineering, db.session, overwrite=True)

    # List the columns in the DB table
    columns = FeatureEngineering.__table__.columns.values()
    column_names = list()
    for column in columns:
        if not column.primary_key:
            column_names.append(column.name)

    logger.info(f'Selected the columns {column_names} to be written to the DB')
    return column_names


@beartype
def store_features_in_db(data: pd.DataFrame, column_names: list, session: Session):
    logger.info(f"Storing {data.shape[0]} rows to '{FEATURE_ENGINEERING_TABLE_NAME}' ...")

    # Select columns in the DataFrame which also appear in the DB table
    data_to_store = (data[column_names]
                     .sort_values(REVISION_ID_COL, ascending=True)
                     .to_dict(orient='records'))

    insert_statement = FeatureEngineering.__table__.insert().values(data_to_store)
    session.execute(insert_statement)
    session.commit()


def store_features_in_db_chunks(data: pd.DataFrame, column_names: list, session: Session):
    logger.info(f"Storing {data.shape[0]} rows to '{FEATURE_ENGINEERING_TABLE_NAME}' ...")

    # Select columns in the DataFrame which also appear in the DB table
    data_to_store = (data[column_names]
                     .sort_values(REVISION_ID_COL, ascending=True)
                     .to_dict(orient='records'))

    insert_statement = data_to_store.to_sql('users', con=Session, if_exists='replace', chunksize=100)
    # insert_statement = FeatureEngineering.__table__.insert().values(data_to_store)
    session.execute(insert_statement)
    session.commit()
