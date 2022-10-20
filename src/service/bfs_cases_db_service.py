import boto3
import pandas as pd
from beartype import beartype
from decouple import config
from loguru import logger
from pandas import DataFrame
from sqlalchemy import create_engine, event, func
from sqlalchemy.orm import sessionmaker

from src.models.BfsCase import BfsCase
from src.models.Clinic import Clinic
from src.models.Hospital import Hospital
from src.models.Revision import Revision
from src.models.chop_code import ChopCode
from src.models.icd_code import IcdCode
from src.revised_case_normalization.py.global_configs import AIMEDIC_ID_COL

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


def get_hospital_cases_df(hopsital_name) -> DataFrame:
    """

    @param hopsital_name:

    @return:
    """
    query = session.query(BfsCase).join(Hospital).filter(Hospital.name == hopsital_name)
    return pd.read_sql(query.statement, session.bind)


def get_clinics():
    return session.query(Clinic).all()


@beartype
def get_sociodemographics_for_hospital_year(hospital_name: str, year: int) -> pd.DataFrame:
    """
    Get the cases filtered by year and hospital name, joint together with all its ICD and CHOP codes.
    @param hospital_name:
    @param year:
    @return: a Dataframe with all matching cases.
    """
    query_sociodemo = (
        session
        .query(BfsCase)
        # TODO: Get the list of entities to select from a list of strings, which is in the VALIDATION_COLS list
        .with_entities(BfsCase.aimedic_id,
                       BfsCase.case_id,
                       BfsCase.patient_id,
                       BfsCase.gender,
                       BfsCase.age_years,
                       BfsCase.duration_of_stay)
        .join(Hospital, BfsCase.hospital_id == Hospital.hospital_id)
        .filter(Hospital.name == hospital_name)
        .filter(BfsCase.discharge_year == year)
        .filter(BfsCase.case_id != '')
    )

    df = pd.read_sql(query_sociodemo.statement, session.bind)

    num_cases_in_db = df.shape[0]
    if num_cases_in_db == 0:
        raise ValueError(f"There is no data for the hospital '{hospital_name}' in {year}")
    else:
        logger.info(f"Read {num_cases_in_db} rows from the DB, for the hospital '{hospital_name}' in {year}")

    return df


@beartype
def get_earliest_revisions_for_aimedic_ids(aimedic_ids: list[int]) -> pd.DataFrame:
    # SELECT
    #     array_agg(revision_date) as revision_date,
    #     array_agg(revision_id) as revision_id
    # FROM coding_revision.revisions
    # WHERE aimedic_id = 120078
    # GROUP BY aimedic_id;

    query_revisions = (
        session
        .query(
            # func.first(Revision.aimedic_id).label(AIMEDIC_ID_COL),
            func.array_agg(Revision.revision_date).label('revision_date'),
            func.array_agg(Revision.revision_id).label('revision_id'),
        )
        .filter(Revision.aimedic_id.in_(aimedic_ids))
        .group_by(Revision.aimedic_id)
    )

    df = pd.read_sql(query_revisions.statement, session.bind)

    # TODO [P1]: Pull aimedic_id together with info above
    # TODO [P2]: Pair / zip revision_date and revision_id per row, and select the revision_id for the earliest revision_date



    return df


# TODO Remove this function after merging pull request #4
@beartype
def get_hospital_year_cases(hospital_name: str, year: int) -> pd.DataFrame:
    """
    Get the cases filtered by year and hospital name, joint together with all its ICD and CHOP codes.
    @param hospital_name:
    @param year:
    @return: a Dataframe with all matching cases.
    """
    subquery_sociodemo = (
        session
        .query(BfsCase)
        .join(Hospital, BfsCase.hospital_id == Hospital.hospital_id)
        .filter(Hospital.name == hospital_name)
        .filter(BfsCase.discharge_year == year)
        .subquery()
    )

    # df = pd.read_sql(subquery_sociodemo.statement, session.bind)

    print("")

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

    subquery_bfs_icds = session.query(subquery_sociodemo,
                                      subquery_icds.c.icds,
                                      subquery_icds.c.icds_ccl,
                                      subquery_icds.c.icds_is_primary,
                                      subquery_icds.c.icds_is_grouper_relevant
                                      ).join(subquery_icds,
                                             subquery_sociodemo.c.aimedic_id == subquery_icds.c.aimedic_id,
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
