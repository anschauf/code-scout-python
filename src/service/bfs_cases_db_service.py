import boto3
import numpy as np
import pandas as pd
from beartype import beartype
from decouple import config
from loguru import logger
from pandas import DataFrame
from sqlalchemy import create_engine, event, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql.elements import Null
from sqlalchemy import tuple_
from src.models.clinic import Clinic
from src.models.procedures import Procedures
from src.models.diagnoses import Diagnoses
from src.models.hospital import Hospital
from src.models.revisions import Revisions
from src.models.sociodemographics import Sociodemographics
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

engine = create_engine(f'postgresql://{BFS_CASES_DB_USER}@{BFS_CASES_DB_URL}:{BFS_CASES_DB_PORT}/{BFS_CASES_DB_NAME}')


@event.listens_for(engine, "do_connect")
def receive_do_connect(dialect, conn_rec, cargs, cparams):
    token = client.generate_db_auth_token(DBHostname=BFS_CASES_DB_URL, Port=BFS_CASES_DB_PORT,
                                          DBUsername=BFS_CASES_DB_USER, Region=AWS_REGION)
    cparams["password"] = token


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

    query = session.query(Sociodemographics).filter(Sociodemographics.case_id.in_(case_ids))

    return pd.read_sql(query.statement, session.bind)


def get_hospital_cases_df(hopsital_name) -> DataFrame:
    """

    @param hopsital_name:

    @return:
    """
    query = (session.query(Sociodemographics)
             .join(Hospital)
             .filter(Hospital.name == hopsital_name))
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
        .query(Sociodemographics)
        # TODO: Get the list of entities to select from a list of strings, which is in the VALIDATION_COLS list
        .with_entities(Sociodemographics.aimedic_id,
                       Sociodemographics.case_id,
                       Sociodemographics.patient_id,
                       Sociodemographics.age_years,
                       Sociodemographics.age_days,
                       Sociodemographics.admission_weight,
                       Sociodemographics.gestation_age,
                       Sociodemographics.gender,
                       Sociodemographics.admission_date,
                       Sociodemographics.grouper_admission_type,
                       Sociodemographics.discharge_date,
                       Sociodemographics.grouper_discharge_type,
                       Sociodemographics.duration_of_stay,
                       Sociodemographics.ventilation_hours)
        .join(Hospital, Sociodemographics.hospital_id == Hospital.hospital_id)
        .filter(Hospital.name == hospital_name)
        .filter(Sociodemographics.discharge_year == year)
        .filter(Sociodemographics.case_id != '')
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
            func.array_agg(Revisions.aimedic_id).label(AIMEDIC_ID_COL),
            func.array_agg(Revisions.revision_date).label(REVISION_DATE_COL),
            func.array_agg(Revisions.revision_id).label(REVISION_ID_COL),
        )
        .filter(Revisions.aimedic_id.in_(aimedic_ids))
        .group_by(Revisions.aimedic_id)
    )

    df = pd.read_sql(query_revisions.statement, session.bind)

    def get_earliest_revision_id(row):
        aimedic_id = row[AIMEDIC_ID_COL][0]  # pick only the first one as they are all the same (because of the group-by)

        revision_dates = row[REVISION_DATE_COL]
        n_revisions = len(revision_dates)
        if n_revisions == 1:
            min_idx = 0  # the only item
        else:
            min_idx = np.argmin(revision_dates)

        row[AIMEDIC_ID_COL] = aimedic_id
        row[REVISION_ID_COL] = row[REVISION_ID_COL][min_idx]
        return row

    df = df.apply(get_earliest_revision_id, axis=1)
    df.drop(columns=[REVISION_DATE_COL], inplace=True)
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
        .query(Sociodemographics)
        .join(Hospital, Sociodemographics.hospital_id == Hospital.hospital_id)
        .filter(Hospital.name == hospital_name)
        .filter(Sociodemographics.discharge_year == year)
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
def insert_revised_cases_into_revisions(revised_case_revision_df: pd.DataFrame) -> dict:
    """Insert revised cases into table coding_revision.revisions.
    @param revised_case_revision_df: a Dataframe of revised case after grouping
    @return: a dictionary with aimedic_id as keys and revision_id as values created after insert into DB.
    """
    logger.info(f"Trying to insert {revised_case_revision_df.shape[0]} cases into the 'Revisions' table ...")
    revision_list = revised_case_revision_df.to_dict(orient='records')

    values_to_insert = list()

    for revision in revision_list:
        values_to_insert.append({
            "aimedic_id": int(revision[AIMEDIC_ID_COL]),
            "drg": str(revision[DRG_COL]),
            "drg_cost_weight": float(revision[DRG_COST_WEIGHT_COL]),
            "effective_cost_weight": float(revision[EFFECTIVE_COST_WEIGHT_COL]),
            "pccl": int(revision[PCCL_COL]),
            "revision_date": str(revision[REVISION_DATE_COL])
        })

    values_info = [(values_dict["aimedic_id"], values_dict['revision_date']) for values_dict in values_to_insert]

    num_rows_before = session.query(Revisions).count()
    delete_statement = (Revisions.__table__
                        .delete()
                        .where(tuple_(Revisions.aimedic_id, Revisions.revision_date).in_(values_info)))
    session.execute(delete_statement)
    session.commit()

    num_rows_after = session.query(Revisions).count()
    if num_rows_after != num_rows_before:
        logger.info(f'Deleted {num_rows_before - num_rows_after} rows from the "Revisions" table, which is about to be updated')

    insert_statement = (Revisions.__table__
                        .insert()
                        .values(values_to_insert)
                        .returning(Revisions.aimedic_id, Revisions.revision_id))

    result = session.execute(insert_statement).fetchall()
    session.commit()

    aimedic_id_with_revision_id = {aimedic_id: revision_id for aimedic_id, revision_id in result}
    logger.success(f"Inserted {len(result)} cases into the 'Revisions' table")
    return aimedic_id_with_revision_id


@beartype
def insert_revised_cases_into_diagnoses(revised_case_diagnoses: pd.DataFrame, aimedic_id_with_revision_id: dict):
    """
    Insert revised cases into table coding_revision.diagnoses.
    @param revised_case_diagnoses: a Dataframe of revised case for diagnoses after grouping.
    @param aimedic_id_with_revision_id: a dictionary with aimedic_id as keys and revision_id as values which are created
        after insert into DB.
    """
    logger.info(f"Trying to insert {revised_case_diagnoses.shape[0]} rows into the 'Diagnoses' table ...")

    diagnosis_list = revised_case_diagnoses.to_dict(orient='records')

    values_to_insert = list()

    for diagnoses in diagnosis_list:
        aimedic_id = int(diagnoses[AIMEDIC_ID_COL])

        values_to_insert.append({
            "aimedic_id": aimedic_id,
            "revision_id": int(aimedic_id_with_revision_id[aimedic_id]),
            "code": str(diagnoses[CODE_COL]),
            "ccl": int(diagnoses[CCL_COL]),
            "is_primary": bool(diagnoses[IS_PRIMARY_COL]),
            "is_grouper_relevant": bool(diagnoses[IS_GROUPER_RELEVANT_COL])
        })

    values_info = [(values_dict["aimedic_id"], values_dict['revision_id']) for values_dict in values_to_insert]

    num_rows_before = session.query(Diagnoses).count()
    delete_statement = (Diagnoses.__table__
                        .delete()
                        .where(tuple_(Diagnoses.aimedic_id, Diagnoses.revision_id).in_(values_info)))
    session.execute(delete_statement)
    session.commit()

    num_rows_after = session.query(Diagnoses).count()
    if num_rows_after != num_rows_before:
        logger.info(f'Deleted {num_rows_before - num_rows_after} rows from the "Diagnoses" table, which is about to be updated')

    insert_statement = (Diagnoses.__table__
                        .insert()
                        .values(values_to_insert))

    session.execute(insert_statement)
    session.commit()

    logger.success(f"Inserted {len(values_to_insert)} rows into the 'Diagnoses' table")


@beartype
def insert_revised_cases_into_procedures(revised_case_procedures: pd.DataFrame, aimedic_id_with_revision_id: dict):
    """Insert revised cases into table coding_revision.procedures.

    @param revised_case_procedures: a Dataframe of revised case for procedures after grouping.
    @param aimedic_id_with_revision_id: a dictionary with aimedic_id as keys and revision_id as values which are created after insert into DB
    """
    logger.info(f"Trying to insert {revised_case_procedures.shape[0]} rows into the 'Procedures' table ...")

    procedure_list = revised_case_procedures.to_dict(orient='records')

    values_to_insert = list()

    for procedure in procedure_list:
        aimedic_id = int(procedure[AIMEDIC_ID_COL])

        # Get the procedure date as None or as a string
        procedure_date = procedure[PROCEDURE_DATE_COL]
        if procedure_date is None or isinstance(procedure_date, Null):
            procedure_date = None
        else:
            procedure_date = str(procedure_date)

        values_to_insert.append({
            "aimedic_id": aimedic_id,
            "revision_id": int(aimedic_id_with_revision_id[aimedic_id]),
            "code": str(procedure[CODE_COL]),
            "side": str(procedure[PROCEDURE_SIDE_COL]),
            "date": procedure_date,
            "is_primary": bool(procedure[IS_PRIMARY_COL]),
            "is_grouper_relevant": bool(procedure[IS_GROUPER_RELEVANT_COL])
        })

    values_info = [(values_dict["aimedic_id"], values_dict['revision_id']) for values_dict in values_to_insert]

    num_rows_before = session.query(Procedures).count()
    delete_statement = (Procedures.__table__
                        .delete()
                        .where(tuple_(Procedures.aimedic_id, Procedures.revision_id).in_(values_info)))
    session.execute(delete_statement)
    session.commit()

    num_rows_after = session.query(Procedures).count()
    if num_rows_after != num_rows_before:
        logger.info(f"Deleted {num_rows_before - num_rows_after} rows from the 'Procedures' table, which is about to be updated")

    insert_statement = (Procedures.__table__
                        .insert()
                        .values(values_to_insert))

    session.execute(insert_statement)
    session.commit()

    logger.success(f"Inserted {len(values_to_insert)} rows into the 'Procedures' table")
