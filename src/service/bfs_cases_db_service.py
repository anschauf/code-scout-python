import boto3
import pandas as pd
import sqlalchemy
import numpy as np
from beartype import beartype
from decouple import config
from loguru import logger
from pandas import DataFrame
from sqlalchemy import create_engine, event, func, MetaData, Table
from sqlalchemy.orm import sessionmaker, declarative_base

from src.models.BfsCase import BfsCase
from src.models.Clinic import Clinic
from src.models.Hospital import Hospital
# from src.models.Revision import Revision
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


# Create a Revision class for table revision from Database
metadata_obj = MetaData(schema="coding_revision")
Base = declarative_base()
class Revision(Base):
    __table__ = Table(
        "revisions",
        metadata_obj,
        autoload_with=engine
    )

# Create a Diagnoses class for table diagnoses from Database
class Diagnoses(Base):
    __table__ = Table(
        "diagnoses",
        metadata_obj,
        autoload_with=engine
    )
# Create a Procedures class for table procedures from Database
class Procedures(Base):
    __table__ = Table(
        "procedures",
        metadata_obj,
        autoload_with=engine
    )

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
                       BfsCase.age_years,
                       BfsCase.age_days,
                       BfsCase.admission_weight,
                       BfsCase.gestation_age,
                       BfsCase.gender,
                       BfsCase.admission_date,
                       BfsCase.grouper_admission_type,
                       BfsCase.discharge_date,
                       BfsCase.grouper_discharge_type,
                       BfsCase.duration_of_stay,
                       BfsCase.ventilation_hours)
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
        .with_entities(Procedures.aimedic_id, Procedures.revision_id, Procedures.code, Procedures.side, Procedures.date, Procedures.is_primary)
        .filter(Procedures.aimedic_id.in_(all_aimedic_ids))
        .filter(Procedures.revision_id.in_(all_revision_ids))
    )

    df = pd.read_sql(query_procedures.statement, session.bind)

    # Select a subset of rows, which contain the primary procedures for each case in grouper format

    primary_procedure = df[df['is_primary']][['revision_id', 'code', 'side', 'date']].apply(list)
    primary_procedure['date'] = primary_procedure['date'].astype(str)
    primary_procedure['date'] = primary_procedure['date'].str.replace("-","")

    primary_procedure['primary_procedure'] = primary_procedure['code'].map(str)+":" + primary_procedure['side'].map(str) + ":" + primary_procedure['date'].map(str)

    primary_procedure = primary_procedure.drop(columns=['code','side', 'date'])

    primary_procedure['primary_procedure'] = primary_procedure['primary_procedure'].str.replace(" ", "")

    # Select a subset of rows, which contain the secondary procedures for each case in grouper format

    secondary_procedures = df[~df['is_primary']][['revision_id', 'code', 'side', 'date']]

    # Format date of secondary procedures
    secondary_procedures['date'] = secondary_procedures['date'].astype(str)
    secondary_procedures['date'] = secondary_procedures['date'].str.replace("-", "")

    # Concatinating columns of secondary procedures to match goruper format
    secondary_procedures[SECONDARY_PROCEDURES_COL] = secondary_procedures['code'].map(str) + ":" + \
                                                             secondary_procedures['side'].map(str) + ":" + \
                                                             secondary_procedures['date'].map(str)

    secondary_procedures[SECONDARY_PROCEDURES_COL] = secondary_procedures[SECONDARY_PROCEDURES_COL].str.replace(" ","")
    secondary_procedures = secondary_procedures.drop(columns=['code', 'side', 'date'])
    secondary_procedures = secondary_procedures.groupby('revision_id', group_keys=True)[SECONDARY_PROCEDURES_COL].apply(list)

    codes_df = (df_revision_ids
                .merge(primary_procedure, on='revision_id', how='left')
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

        if isinstance(row[SECONDARY_DIAGNOSES_COL], float):
            row[SECONDARY_DIAGNOSES_COL] = list()

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


@beartype
def insert_revised_case_into_revisions(revised_case_revision_df: pd.DataFrame) -> dict:
    """
    Insert revised cases into table code_revision.revisions
    @param revised_case_revision_df: a Dataframe of revised case after Grupper
    Columns needed in the revised_case DataFrame:
    aimedic_id, drg, drg_cost_weight, effective_cost_weight, pccl, revision_date,

    @return: a dictionary with aimedic_id as keys and revision_id as values created after insert into DB.
    """
    # columns for table revision
    # revision_id: auto increment
    # aimedic_id, drg, adrg, drg_cost_weight, effective_cost_weight, pccl
    # revision_date: not available yet
    revision_list = revised_case_revision_df.to_dict(orient='records')
    aimedic_id_with_revision_id = dict()
    for revision in revision_list:
        aimedic_id = revision['aimedic_id']
        drg = revision['drg']
        drg_cost_weight = revision['drg_cost_weight']
        effective_cost_weight = revision['effective_cost_weight']
        pccl = revision['pccl']
        revision_date = revision['revision_date']
        revision_obj = Revision(aimedic_id=aimedic_id, drg=drg, drg_cost_weight=drg_cost_weight, effective_cost_weight=effective_cost_weight, pccl=pccl, revision_date=revision_date) #change data to Revision object
        session.add(revision_obj)
        session.flush() # push the insert data into DB
        # session.refresh(revision_obj) might need to be refreshed first
        revision_id = revision_obj.revision_id # get the created primary key: revision_id
        aimedic_id_with_revision_id[aimedic_id] = revision_id
    session.commit() # save change to DB

    return aimedic_id_with_revision_id

@beartype
def insert_revised_case_into_diagonoses(revised_case_diagonoses:pd.DataFrame, aimedic_id_with_revision_id: dict) -> None:
    """
    Insert revised cases into table code_revision.diagonoses
    @param
    revised_case_diagonoses: a Dataframe of revised case for diagonoses after Grupper.
    aimedic_id_with_revision_id: a dictionary with aimedic_id as keys and revision_id as values created after insert into DB.

    @return: None
    """
    # columns for table revision
    # diagnoses_pk:  auto-increment
    # aimedic_id, revision_id, code, is_primary,
    # ccl, is_grouper_relevant: from Grupper
    # prepare the format for diagonoses table
    insert_col = ['aimedic_id', 'revision_id', 'code', 'ccl', 'is_primary', 'is_grouper_relevant']
    revised_case_diagonoses['revision_id'] = revised_case_diagonoses['aimedic_id'].map(aimedic_id_with_revision_id)

    revision_diagonoses_list = revised_case_diagonoses.to_dict(orient='records')

    revision_diagonoses_obj = [Diagnoses(revision) for revision in revision_diagonoses_list]
    # if we do not need to save diagnoses_id, all records can be added using add_all
    session.add_all(revision_diagonoses_obj)
    session.commit()


@beartype
def insert_revised_case_into_procedures(revised_case_with_revision_id: pd.DataFrame) -> None:
    """
    Insert revised cases into table code_revision.procedures
    @param revised_case_with_revision_id: a Dataframe of revised case with added revision_id after insert into revision.

    @return: None
    """

    # columns for table revision
    # procedures_pk:  auto-increment
    # aimedic_id, revision_id, code, is_primary,
    # ccl, is_grouper_relevant: from Grupper
    # side, date,
    # is_grouper_relevant: from Grupper
    insert_col = ['aimedic_id', 'revision_id', 'code', 'side','date',  'is_grouper_relevant', 'is_primary'] # TODO: check the sequence and name in the DB again
    revision_procedures = revised_case_with_revision_id[insert_col]

    revision_procedures_list = revision_procedures.to_dict(orient='records')
    revision_procedures_obj = [Diagnoses(revision) for revision in revision_procedures_list]
    # if we do not need to save procedures_id, all records can be added using add_all
    session.add_all(revision_procedures_obj)
    session.commit()






