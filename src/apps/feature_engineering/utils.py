import math
from typing import Optional

import pandas as pd
from beartype import beartype
from loguru import logger
from sqlalchemy.orm import Session

from src.data_model.feature_engineering import FeatureEngineering
from src.data_model.sociodemographics import Sociodemographics
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
def store_features_in_db(data: pd.DataFrame, chunksize: int, session: Session):
    n_rows = data.shape[0]
    n_chunks = math.ceil(float(n_rows) / chunksize)

    logger.info(f"Storing {n_rows} rows to '{FEATURE_ENGINEERING_TABLE_NAME}', in {n_chunks} of {chunksize} rows at a time ...")

    connection = session.connection(execution_options={'stream_results': True})

    num_rows_inserted = pd.to_sql(
            name=FeatureEngineering.__tablename__, schema=FeatureEngineering.__table__.schema,
            con=connection, if_exists='append', index=False, method='multi',
            chunksize=chunksize
    )

    logger.success(f'Inserted {num_rows_inserted} rows')
