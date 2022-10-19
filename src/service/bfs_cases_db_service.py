import boto3
import pandas as pd
from decouple import config
from pandas import DataFrame
from sqlalchemy import create_engine, event, extract, func
from sqlalchemy.orm import sessionmaker

from src.models.BfsCase import BfsCase
from src.models.Clinic import Clinic
from src.models.Hospital import Hospital
from src.models.chop_code import ChopCode
from src.models.icd_code import IcdCode

# import envs
BFS_CASES_DB_URL = config('BFS_CASES_DB_URL')
BFS_CASES_DB_USER = config('BFS_CASES_DB_USER')
BFS_CASES_DB_NAME = config('BFS_CASES_DB_NAME')
BFS_CASES_DB_PORT = config('BFS_CASES_DB_PORT')
AWS_REGION = config('AWS_REGION')
AWS_ACCESS_KEY_ID = config('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = config('AWS_SECRET_ACCESS_KEY')

# gets the credentials from .aws/credentials
client = boto3.client('rds', region_name=AWS_REGION, aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

engine = create_engine(
    f'postgresql://{BFS_CASES_DB_USER}@{BFS_CASES_DB_URL}:{BFS_CASES_DB_PORT}/{BFS_CASES_DB_NAME}')


@event.listens_for(engine, "do_connect")
def receive_do_connect(dialect, conn_rec, cargs, cparams):
    token = client.generate_db_auth_token(DBHostname=BFS_CASES_DB_URL, Port=BFS_CASES_DB_PORT,
                                          DBUsername=BFS_CASES_DB_USER, Region=AWS_REGION)
    cparams["password"] = token
    # return psycopg2.connect(host=BFS_CASES_DB_URL, port=BFS_CASES_DB_PORT, database=BFS_CASES_DB_NAME,
    #                         user=BFS_CASES_DB_USER, password=token)


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
    return pd.read_sql(sql_query, session.bind)


def get_bfs_cases_by_ids(case_ids: list) -> DataFrame:
    """

    @param case_ids:
    @return:
    """

    query = session.query(BfsCase).filter(BfsCase.case_id.in_(case_ids))

    return pd.read_sql(query.statement, session.bind)


def get_bfs_cases_by_ids_no_pad0(case_ids: list) -> DataFrame:
    """

    @param case_ids:
    @return:
    """
    cases = list()
    for case_id in case_ids:
        case_id_pad = '%' + case_id
        query = session.query(BfsCase).filter(BfsCase.case_id.like(case_id_pad))
        cases.append(pd.read_sql(query.statement, session.bind))

    return pd.concat(cases)


def get_hospital_cases_df(hopsital_name) -> DataFrame:
    """

    @param hopsital_name:

    @return:
    """
    query = session.query(BfsCase).join(Hospital).filter(Hospital.name == hopsital_name)
    return pd.read_sql(query.statement, session.bind)


def get_clinics():
    return session.query(Clinic).all()


# TODO Remove this function after merging pull request #4
def get_hospital_year_cases(hospital_name, year):
    """
    Get the cases filtered by year and hospital name, joint together with all its ICD and CHOP codes.
    @param hospital_name:
    @param year:
    @return: a Dataframe with all matching cases.
    """
    subquery_cases_from_hospital_year = session.query(BfsCase).join(Hospital).filter(
        Hospital.name == hospital_name).filter(extract('year', BfsCase.discharge_date) == year).subquery()

    subquery_icds = session.query(IcdCode.aimedic_id,
                                  func.array_agg(IcdCode.code).label('icds'),
                                  func.array_agg(IcdCode.ccl).label('icds_ccl'),
                                  func.array_agg(IcdCode.is_primary).label('icds_is_primary'),
                                  func.array_agg(IcdCode.is_grouper_relevant).label('icds_is_grouper_relevant')
                                  ).group_by(IcdCode.aimedic_id).subquery()

    subquery_chops = session.query(ChopCode.aimedic_id,
                                   func.array_agg(ChopCode.code).label('chops'),
                                   func.array_agg(ChopCode.side).label('chops_side'),
                                   func.array_agg(ChopCode.date).label('chops_date'),
                                   func.array_agg(ChopCode.is_grouper_relevant).label('chops_is_grouper_relevant'),
                                   func.array_agg(ChopCode.is_primary).label('chops_is_primary'),
                                   ).group_by(ChopCode.aimedic_id).subquery()

    subquery_bfs_icds = session.query(subquery_cases_from_hospital_year,
                                      subquery_icds.c.icds,
                                      subquery_icds.c.icds_ccl,
                                      subquery_icds.c.icds_is_primary,
                                      subquery_icds.c.icds_is_grouper_relevant
                                      ).join(subquery_icds,
                                             subquery_cases_from_hospital_year.c.aimedic_id == subquery_icds.c.aimedic_id,
                                             isouter=True
                                             ).subquery()

    query = session.query(subquery_bfs_icds,
                          subquery_chops.c.chops,
                          subquery_chops.c.chops_side,
                          subquery_chops.c.chops_date,
                          subquery_chops.c.chops_is_grouper_relevant,
                          subquery_chops.c.chops_is_primary
                          ).join(subquery_chops,
                                 subquery_bfs_icds.c.aimedic_id == subquery_chops.c.aimedic_id, isouter=True)

    return pd.read_sql(query.statement, session.bind)
