from decouple import config
import pandas as pd
from pandas import DataFrame

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.models.Clinic import Clinic
from src.models.Hospital import Hospital
from src.models.BfsCase import BfsCase
from src.models.icd_code import IcdCode

# import envs
BFS_CASES_DB_URL = config('BFS_CASES_DB_URL')
BFS_CASES_DB_USER = config('BFS_CASES_DB_USER')
BFS_CASES_DB_NAME = config('BFS_CASES_DB_NAME')
BFS_CASES_DB_PASSWORD = config('BFS_CASES_DB_PASSWORD')
BFS_CASES_DB_PORT = config('BFS_CASES_DB_PORT')

engine = create_engine(
    f'postgresql://{BFS_CASES_DB_USER}:{BFS_CASES_DB_PASSWORD}@{BFS_CASES_DB_URL}:{BFS_CASES_DB_PORT}/{BFS_CASES_DB_NAME}')

# create a configured "Session" class
Session = sessionmaker(bind=engine)

# create a Session
session = Session()

def get_by_sql_query(sql_query) -> DataFrame:
    """
    Make any SQL request for the BFS-cases DB.
    @param sql_query: the sql query as String
    @return: matches as DataFrame.
    """
    return pd.read_sql(sql_query, con=engine)


def get_hospital_cases_db(hopsital_name) -> DataFrame:
    """

    @param hopsital_name:
    @return:
    """
    query = session.query(BfsCase).join(Hospital).filter(Hospital.name == hopsital_name)
    return pd.read_sql(query.statement, session.bind)

def get_clinics() -> [Clinic]:
    return session.query(Clinic).all()
def get_test():
    return pd.read_sql(session.query(IcdCode).limit(10).statement, session.bind)
    # return session.query(IcdCode).all()