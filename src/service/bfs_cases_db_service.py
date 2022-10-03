from decouple import config
import pandas as pd
from pandas import DataFrame

from sqlalchemy import create_engine

# import envs
BFS_CASES_DB_URL = config('BFS_CASES_DB_URL')
BFS_CASES_DB_USER = config('BFS_CASES_DB_USER')
BFS_CASES_DB_NAME = config('BFS_CASES_DB_NAME')
BFS_CASES_DB_PASSWORD = config('BFS_CASES_DB_PASSWORD')
BFS_CASES_DB_PORT = config('BFS_CASES_DB_PORT')

engine = create_engine(
    f'postgresql://{BFS_CASES_DB_USER}:{BFS_CASES_DB_PASSWORD}@{BFS_CASES_DB_URL}:{BFS_CASES_DB_PORT}/{BFS_CASES_DB_NAME}')


def get_by_sql_query(sql_query) -> DataFrame:
    """
    Make any SQL request for the BFS-cases DB.
    @param sql_query: the sql query as String
    @return: matches as DataFrame.
    """
    return pd.read_sql(sql_query, con=engine)
