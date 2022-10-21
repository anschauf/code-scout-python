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
from src.models.chop_code import Procedures
from src.models.icd_code import Diagnoses
from src.revised_case_normalization.py.global_configs import *

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
    query_revisions = (
        session
        .query(
            func.array_agg(Revision.aimedic_id).label(AIMEDIC_ID_COL),
            func.array_agg(Revision.revision_date).label('revision_date'),
            func.array_agg(Revision.revision_id).label('revision_id'),
        )
        .filter(Revision.aimedic_id.in_(aimedic_ids))
        .group_by(Revision.aimedic_id)
    )

    df = pd.read_sql(query_revisions.statement, session.bind)

    def get_earliest_revision_id(row):
        aimedic_id = row[AIMEDIC_ID_COL][0]  # pick only the first one as they are all the same (because of the group-by)

        n_revisions = len(row['revision_date'])
        if n_revisions == 1:
            revision_id = row['revision_id'][0]
        else:
            raise NotImplementedError("Don't know what to do when there are mulitple revisions")

        row[AIMEDIC_ID_COL] = aimedic_id
        row['revision_id'] = revision_id
        return row

    df = df.apply(get_earliest_revision_id, axis=1)
    df.drop(columns=['revision_date'], inplace=True)
    return df


@beartype
def get_diagnoses_codes(df_revision_ids: pd.DataFrame) -> pd.DataFrame:
    all_aimedic_ids = set(df_revision_ids[AIMEDIC_ID_COL].values.tolist())
    all_revision_ids = set(df_revision_ids['revision_id'].values.tolist())

    query_diagnoses = (
        session
        .query(Diagnoses)
        .with_entities(Diagnoses.aimedic_id, Diagnoses.revision_id, Diagnoses.code, Diagnoses.is_primary)
        .filter(Diagnoses.aimedic_id.in_(all_aimedic_ids))
        .filter(Diagnoses.revision_id.in_(all_revision_ids))
    )

    df = pd.read_sql(query_diagnoses.statement, session.bind)

    # Select a subset of rows, which contain the primary diagnosis for each case
    primary_diagnoses = df[df['is_primary']][['revision_id', 'code']]
    primary_diagnoses.rename(columns={'code': PRIMARY_DIAGNOSIS_COL}, inplace=True)

    # Aggregate the subset of rows, which contain the secondary diagnoses for each case
    secondary_diagnoses = df[~df['is_primary']].groupby('revision_id', group_keys=True)['code'].apply(list)
    secondary_diagnoses = secondary_diagnoses.to_frame(SECONDARY_DIAGNOSES_COL)
    secondary_diagnoses.reset_index(drop=False, inplace=True)

    codes_df = (df_revision_ids
                .merge(primary_diagnoses, on='revision_id', how='left')
                .merge(secondary_diagnoses, on='revision_id', how='left'))

    n_cases_no_pd = codes_df[PRIMARY_DIAGNOSIS_COL].isna().sum()
    if n_cases_no_pd > 0:
        raise ValueError(f'There are {n_cases_no_pd} cases without a Primary Diagnosis')

    # Replace NaNs with an empty list
    codes_df[SECONDARY_DIAGNOSES_COL] = codes_df[SECONDARY_DIAGNOSES_COL].apply(lambda x: x if isinstance(x, list) else [])

    return codes_df


@beartype
def get_procedures_codes(df_revision_ids: pd.DataFrame) -> pd.DataFrame:
    all_aimedic_ids = set(df_revision_ids[AIMEDIC_ID_COL].values.tolist())
    all_revision_ids = set(df_revision_ids['revision_id'].values.tolist())

    query_procedures = (
        session
        .query(Procedures)
        .with_entities(Procedures.aimedic_id, Procedures.revision_id, Procedures.code, Procedures.is_primary)
        .filter(Procedures.aimedic_id.in_(all_aimedic_ids))
        .filter(Procedures.revision_id.in_(all_revision_ids))
    )

    df = pd.read_sql(query_procedures.statement, session.bind)

    # Select a subset of rows, which contain the primary diagnosis for each case
    primary_procedures = df[df['is_primary']][['revision_id', 'code']]
    primary_procedures.rename(columns={'code': PRIMARY_PROCEDURE_COL}, inplace=True)

    # Aggregate the subset of rows, which contain the secondary diagnoses for each case
    secondary_procedures = df[~df['is_primary']].groupby('revision_id', group_keys=True)['code'].apply(list)
    secondary_procedures = secondary_procedures.to_frame(SECONDARY_PROCEDURES_COL)
    secondary_procedures.reset_index(drop=False, inplace=True)

    codes_df = (df_revision_ids
                .merge(primary_procedures, on='revision_id', how='left')
                .merge(secondary_procedures, on='revision_id', how='left'))

    # Replace NaNs with an empty list
    codes_df[SECONDARY_PROCEDURES_COL] = codes_df[SECONDARY_PROCEDURES_COL].apply(lambda x: x if isinstance(x, list) else [])
    codes_df[PRIMARY_PROCEDURE_COL] = codes_df[PRIMARY_PROCEDURE_COL].fillna('')

    return codes_df


@beartype
def get_codes(df_revision_ids: pd.DataFrame) -> pd.DataFrame:
    diagnoses_df = get_diagnoses_codes(df_revision_ids)
    procedures_df = get_procedures_codes(df_revision_ids)

    # Drop the aimedic_id column to avoid adding it with a suffix and having to remove it later
    codes_df = (df_revision_ids
                .merge(diagnoses_df.drop(columns=AIMEDIC_ID_COL), on='revision_id', how='left')
                .merge(procedures_df.drop(columns=AIMEDIC_ID_COL), on='revision_id', how='left'))

    return codes_df


@beartype
def apply_revisions(cases_df: pd.DataFrame, revisions_df: pd.DataFrame) -> pd.DataFrame:
    joined = pd.merge(cases_df, revisions_df, on=AIMEDIC_ID_COL, how='left')

    # Notes:
    # - revision_id is not needed
    # - the old_pd is not needed, the new_pd is the new PD

    # Add & remove ICD codes from the list of secondary diagnoses
    def revise_diagnoses_codes(row):
        revised_codes = list(row[SECONDARY_DIAGNOSES_COL])

        for code_to_add in row[ADDED_ICD_CODES]:
            revised_codes.append(code_to_add)

        for code_to_remove in row[REMOVED_ICD_CODES]:
            revised_codes.remove(code_to_remove)

        row[SECONDARY_DIAGNOSES_COL] = revised_codes
        return row

    # Delete the primary procedure if it was removed
    def revise_primary_procedure_code(row):
        if row[PRIMARY_PROCEDURE_COL] in row[REMOVED_CHOP_CODES]:
            row[PRIMARY_PROCEDURE_COL] = ''

        return row

    # Add & remove CHOP codes from the list of secondary procedures
    def revise_secondary_procedure_codes(row):
        revised_codes = list(row[SECONDARY_PROCEDURES_COL])

        for code_to_add in row[ADDED_CHOP_CODES]:
            revised_codes.append(code_to_add)

        for code_to_remove in row[REMOVED_CHOP_CODES]:
            # We need to check whether the code is present in this list, because it may appear as primary procedure
            if code_to_remove in revised_codes:
                revised_codes.remove(code_to_remove)

        row[SECONDARY_PROCEDURES_COL] = revised_codes
        return row

    # Apply all the revisions
    joined = joined.apply(revise_diagnoses_codes, axis=1)
    joined = joined.apply(revise_primary_procedure_code, axis=1)
    joined = joined.apply(revise_secondary_procedure_codes, axis=1)

    # Select only the columns of interest
    revised_cases = joined[[AIMEDIC_ID_COL, NEW_PRIMARY_DIAGNOSIS_COL, SECONDARY_DIAGNOSES_COL, PRIMARY_PROCEDURE_COL, SECONDARY_PROCEDURES_COL]]

    return revised_cases


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

    subquery_icds = session.query(Diagnoses.aimedic_id,
                                  func.array_agg(Diagnoses.code).label('icds'),
                                  func.array_agg(Diagnoses.ccl).label('icds_ccl'),
                                  func.array_agg(Diagnoses.is_primary).label('icds_is_primary'),
                                  func.array_agg(Diagnoses.is_grouper_relevant).label('icds_is_grouper_relevant')
                                  ).group_by(Diagnoses.aimedic_id).subquery()

    subquery_chops = session.query(Procedures.aimedic_id,
                                   func.array_agg(Procedures.code).label('chops'),
                                   func.array_agg(Procedures.side).label('chops_side'),
                                   func.array_agg(Procedures.date).label('chops_date'),
                                   func.array_agg(Procedures.is_grouper_relevant).label('chops_is_grouper_relevant'),
                                   func.array_agg(Procedures.is_primary).label('chops_is_primary'),
                                   ).group_by(Procedures.aimedic_id).subquery()

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
