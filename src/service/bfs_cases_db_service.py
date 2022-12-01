import sys
from typing import Optional

import numpy as np
import pandas as pd
from beartype import beartype
from loguru import logger
from pandas import DataFrame
from sqlalchemy import func
from sqlalchemy import tuple_
from sqlalchemy.orm import DeclarativeMeta
from sqlalchemy.orm.session import Session
from sqlalchemy.sql.ddl import CreateSchema, DropTable
from sqlalchemy.sql.elements import Null

from src.data_model.clinic import Clinic
from src.data_model.diagnosis import Diagnosis
from src.data_model.hospital import Hospital
from src.data_model.procedure import Procedure
from src.data_model.revision import Revision
from src.data_model.sociodemographics import Sociodemographics
from src.revised_case_normalization.notebook_functions.global_configs import AIMEDIC_ID_COL, REVISION_DATE_COL, \
    REVISION_ID_COL, PRIMARY_DIAGNOSIS_COL, SECONDARY_DIAGNOSES_COL, PROCEDURE_DATE_COL, PROCEDURE_SIDE_COL, CODE_COL, \
    PRIMARY_PROCEDURE_COL, IS_PRIMARY_COL, SECONDARY_PROCEDURES_COL, DRG_COL, DRG_COST_WEIGHT_COL, \
    EFFECTIVE_COST_WEIGHT_COL, PCCL_COL, CCL_COL, IS_GROUPER_RELEVANT_COL, NEW_PRIMARY_DIAGNOSIS_COL
from src.service.database import Database


@beartype
def get_bfs_cases_by_ids(case_ids: list, session: Session) -> DataFrame:
    """
     Get records from case_data.Sociodemographics table using case_id
     @param case_ids: a list of case_id
     @return: a Dataframe
     """
    query = (session
             .query(Sociodemographics)
             .filter(Sociodemographics.case_id.in_(case_ids)))

    return pd.read_sql(query.statement, session.bind)

@beartype
def get_hospital_cases_df(hospital_name: str, session: Session) -> DataFrame:
    """
     Get records from case_data.Sociodemographics table using hospital_name
     @param hospital_name:
     @return: a Dataframe
     """

    query = (session.query(Sociodemographics)
             .join(Hospital)
             .filter(Hospital.hospital_name == hospital_name))
    return pd.read_sql(query.statement, session.bind)

@beartype
def get_clinics(session: Session):
    """
    Get all records from dimension.clinic table
    """
    return session.query(Clinic).all()


@beartype
def get_sociodemographics_for_hospital_year(hospital_name: str, year: int, session: Session) -> pd.DataFrame:
    """
    Get the cases filtered by year and hospital name, joined together with all its ICD and CHOP codes.
    @param hospital_name:
    @param year:
    @return: a dataframe with all matching cases.
    """
    query_sociodemo = (
        session
        .query(Sociodemographics)
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
        .filter(Hospital.hospital_name == hospital_name)
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
def get_earliest_revisions_for_aimedic_ids(aimedic_ids: list[int], session: Session) -> pd.DataFrame:
    """
     Get earliest revisions of aimedic_ids
     @param session: active DB session
     @param aimedic_ids:
     @return: a Dataframe containing aimedic ids, revision ids and revision dates
     """

    query_revisions = (
        session
        .query(
            func.array_agg(Revision.aimedic_id).label(AIMEDIC_ID_COL),
            func.array_agg(Revision.revision_date).label(REVISION_DATE_COL),
            func.array_agg(Revision.revision_id).label(REVISION_ID_COL),
        )
        .filter(Revision.aimedic_id.in_(aimedic_ids))
        .group_by(Revision.aimedic_id)
    )

    df = pd.read_sql(query_revisions.statement, session.bind)

    def get_earliest_revision_id(row):
        row[AIMEDIC_ID_COL] = row[AIMEDIC_ID_COL][0]  # pick only the first one as they are all the same (because of the group-by)
        revision_dates = row[REVISION_DATE_COL]
        min_idx = np.argmin(revision_dates)
        row[REVISION_ID_COL] = row[REVISION_ID_COL][min_idx]
        return row

    df = df.apply(get_earliest_revision_id, axis=1)
    df.drop(columns=[REVISION_DATE_COL], inplace=True)
    return df


@beartype
def get_diagnoses_codes(df_revision_ids: pd.DataFrame, session: Session) -> pd.DataFrame:
    """
     Retrieve primary and secondary diagnoses of the revised cases from the DB.
     @param session: active DB session
     @param df_revision_ids: a Dataframe with aimedic_id and revision_id
     @return: a Dataframe containing revision ids, primary and secondary diagnoses
     """

    all_aimedic_ids = set(df_revision_ids[AIMEDIC_ID_COL].values.tolist())
    all_revision_ids = set(df_revision_ids[REVISION_ID_COL].values.tolist())

    query_diagnoses = (
        session
        .query(Diagnosis)
        .with_entities(Diagnosis.aimedic_id, Diagnosis.revision_id, Diagnosis.code, Diagnosis.is_primary)
        .filter(Diagnosis.aimedic_id.in_(all_aimedic_ids))
        .filter(Diagnosis.revision_id.in_(all_revision_ids))
    )

    df = pd.read_sql(query_diagnoses.statement, session.bind)

    # Select a subset of rows, which contain the primary diagnosis for each case
    primary_diagnoses = df[df[IS_PRIMARY_COL]][[REVISION_ID_COL, CODE_COL]]
    primary_diagnoses.rename(columns={CODE_COL: PRIMARY_DIAGNOSIS_COL}, inplace=True)

    # Aggregate the subset of rows, which contain the secondary diagnoses for each case
    secondary_diagnoses = df[~df[IS_PRIMARY_COL]].groupby(REVISION_ID_COL, group_keys=True)[CODE_COL].apply(list)
    secondary_diagnoses = secondary_diagnoses.to_frame(SECONDARY_DIAGNOSES_COL)
    secondary_diagnoses.reset_index(drop=False, inplace=True)

    codes_df = (df_revision_ids
                .merge(primary_diagnoses, on=REVISION_ID_COL, how='left')
                .merge(secondary_diagnoses, on=REVISION_ID_COL, how='left'))

    n_cases_no_pd = codes_df[PRIMARY_DIAGNOSIS_COL].isna().sum()
    if n_cases_no_pd > 0:
        raise ValueError(f'There are {n_cases_no_pd} cases without a Primary Diagnosis')

    # Replace NaNs with an empty list
    codes_df[SECONDARY_DIAGNOSES_COL] = codes_df[SECONDARY_DIAGNOSES_COL].apply(lambda x: x if isinstance(x, list) else [])

    return codes_df


@beartype
def get_procedures_codes(df_revision_ids: pd.DataFrame, session: Session) -> pd.DataFrame:
    """
     Retrieve primary and secondary procedures of the revised cases from the DB.
     @param session: active DB session
     @param df_revision_ids: a Dataframe with aimedic_id and revision_id
     @return: a dataframe containing revision ids, primary and secondary diagnoses
     """

    all_aimedic_ids = set(df_revision_ids[AIMEDIC_ID_COL].values.tolist())
    all_revision_ids = set(df_revision_ids['revision_id'].values.tolist())

    query_procedures = (
        session
        .query(Procedure)
        .with_entities(Procedure.aimedic_id, Procedure.revision_id, Procedure.code, Procedure.side, Procedure.date, Procedure.is_primary)
        .filter(Procedure.aimedic_id.in_(all_aimedic_ids))
        .filter(Procedure.revision_id.in_(all_revision_ids))
    )

    df = pd.read_sql(query_procedures.statement, session.bind)

    # Concatenate procedure codes, side and date, as it is a standard used by the SwissDRG grouper
    def concatenate_code_info(row):
        side = row[PROCEDURE_SIDE_COL]
        if side == ' ' or side is None:
            side = ''
        info = (row[CODE_COL], side, str(row[PROCEDURE_DATE_COL]).replace('-', ''))
        row[CODE_COL] = ':'.join(info)
        return row

    df = df.apply(concatenate_code_info, axis=1)

    primary_procedure = df[df[IS_PRIMARY_COL]][[REVISION_ID_COL, CODE_COL]]
    primary_procedure.rename(columns={CODE_COL: PRIMARY_PROCEDURE_COL}, inplace=True)

    secondary_procedures = df[~df[IS_PRIMARY_COL]][[REVISION_ID_COL, CODE_COL]]
    secondary_procedures.rename(columns={CODE_COL: SECONDARY_PROCEDURES_COL}, inplace=True)

    secondary_procedures = secondary_procedures.groupby(REVISION_ID_COL, group_keys=True)[SECONDARY_PROCEDURES_COL].apply(list)

    codes_df = (df_revision_ids
                .merge(primary_procedure, on=REVISION_ID_COL, how='left')
                .merge(secondary_procedures, on=REVISION_ID_COL, how='left'))

    # Fill NaNs
    codes_df[PRIMARY_PROCEDURE_COL] = codes_df[PRIMARY_PROCEDURE_COL].fillna('')
    codes_df[SECONDARY_PROCEDURES_COL] = codes_df[SECONDARY_PROCEDURES_COL].apply(lambda x: x if isinstance(x, list) else [])

    return codes_df


@beartype
def get_codes(df_revision_ids: pd.DataFrame, session: Session) -> pd.DataFrame:
    """
    Merging information on the diagnoses and procedures from the DB for usage in the revise function (revise.notebook_functions)
     @param session: active DB session
     @param df_revision_ids: a Dataframe with aimedic_id and revision_id
     @return: a dataframe containing revision ids, diagnoses and procedures
    """

    diagnoses_df = get_diagnoses_codes(df_revision_ids, session)
    procedures_df = get_procedures_codes(df_revision_ids, session)

    # Drop the aimedic_id column to avoid adding it with a suffix and having to remove it later
    codes_df = (df_revision_ids
                .merge(diagnoses_df.drop(columns=AIMEDIC_ID_COL), on='revision_id', how='left')
                .merge(procedures_df.drop(columns=AIMEDIC_ID_COL), on='revision_id', how='left'))

    return codes_df


@beartype
def insert_revised_cases_into_revisions(revised_case_revision_df: pd.DataFrame, session: Session) -> dict:
    """
    Insert revised cases into the table coding_revision.revisions
    @param session: active DB session
    @param revised_case_revision_df: a Dataframe of revised cases after grouping
    @return: a dictionary with aimedic_ids as keys and revision_ids as values created after insert into the DB
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

    num_rows_before = session.query(Revision).count()
    delete_statement = (Revision.__table__
                        .delete()
                        .where(tuple_(Revision.aimedic_id, Revision.revision_date).in_(values_info)))
    session.execute(delete_statement)
    session.commit()

    num_rows_after = session.query(Revision).count()
    if num_rows_after != num_rows_before:
        logger.info(f'Deleted {num_rows_before - num_rows_after} rows from the "Revisions" table, which is about to be updated')

    insert_statement = (Revision.__table__
                        .insert()
                        .values(values_to_insert)
                        .returning(Revision.aimedic_id, Revision.revision_id))

    result = session.execute(insert_statement).fetchall()
    session.commit()

    aimedic_id_with_revision_id = {aimedic_id: revision_id for aimedic_id, revision_id in result}
    logger.success(f"Inserted {len(result)} cases into the 'Revisions' table")
    return aimedic_id_with_revision_id


@beartype
def insert_revised_cases_into_diagnoses(revised_case_diagnoses: pd.DataFrame, aimedic_id_with_revision_id: dict, session: Session):
    """
    Insert revised cases into the table coding_revision.diagnoses
    @param session: active DB session
    @param revised_case_diagnoses: a Dataframe of revised cases for diagnoses after grouping
    @param aimedic_id_with_revision_id: a dictionary with aimedic_ids as keys and revision_ids as values which are created
        after insert into the DB
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

    num_rows_before = session.query(Diagnosis).count()
    delete_statement = (Diagnosis.__table__
                        .delete()
                        .where(tuple_(Diagnosis.aimedic_id, Diagnosis.revision_id).in_(values_info)))
    session.execute(delete_statement)
    session.commit()

    num_rows_after = session.query(Diagnosis).count()
    if num_rows_after != num_rows_before:
        logger.info(f'Deleted {num_rows_before - num_rows_after} rows from the "Diagnoses" table, which is about to be updated')

    insert_statement = (Diagnosis.__table__
                        .insert()
                        .values(values_to_insert))

    session.execute(insert_statement)
    session.commit()

    logger.success(f"Inserted {len(values_to_insert)} rows into the 'Diagnoses' table")


@beartype
def insert_revised_cases_into_procedures(revised_case_procedures: pd.DataFrame, aimedic_id_with_revision_id: dict, session: Session):
    """Insert revised cases into table coding_revision.procedures.

    @param revised_case_procedures: a Dataframe of revised case for procedures after grouping.
    @param aimedic_id_with_revision_id: a dictionary with aimedic_id as keys and revision_id as values which are created
    after insert into the DB.
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

    num_rows_before = session.query(Procedure).count()
    delete_statement = (Procedure.__table__
                        .delete()
                        .where(tuple_(Procedure.aimedic_id, Procedure.revision_id).in_(values_info)))
    session.execute(delete_statement)
    session.commit()

    num_rows_after = session.query(Procedure).count()
    if num_rows_after != num_rows_before:
        logger.info(f"Deleted {num_rows_before - num_rows_after} rows from the 'Procedures' table, which is about to be updated")

    insert_statement = (Procedure.__table__
                        .insert()
                        .values(values_to_insert))

    session.execute(insert_statement)
    session.commit()

    logger.success(f"Inserted {len(values_to_insert)} rows into the 'Procedures' table")


# noinspection PyUnresolvedReferences
@beartype
def create_table(table: DeclarativeMeta, session: Session, *, overwrite: bool = False):
    # We set the isolation level to autocommit, so we don't have to do it after each execute()
    connection = session.connection(execution_options={
        'isolation_level': 'AUTOCOMMIT'
    })

    try:
        schema_name = table.__table__.schema
        table_name = table.__tablename__

        # Create the schema if it doesn't exist
        if not connection.dialect.has_schema(connection, schema_name):
            logger.info(f"Creating the database schema '{schema_name}' ...")
            connection.execute(CreateSchema(schema_name))
            logger.success(f"Created the database schema '{schema_name}'")

        # If the table needs to be overwritten, drop it first
        if overwrite and connection.dialect.has_table(connection, table_name, schema_name):
            logger.info(f"Dropping the table '{schema_name}.{table_name}' ...")
            connection.execute(DropTable(table.__table__, if_exists=True))
            logger.info(f"Dropped the table '{schema_name}.{table_name}'")

        # Create the table
        table.metadata.create_all(connection, checkfirst=True)
        logger.success(f"Created the table '{schema_name}.{table_name}'")

    except Exception as e:
        logger.error(e)

    finally:
        connection.close()


@beartype
def read_cases(*,
               n_rows: Optional[int] = None
               ) -> pd.DataFrame:
    with Database() as db:
        socio_query = (db.session
         .query(Sociodemographics, Hospital, Clinic, Revision)
         .join(Hospital, Sociodemographics.hospital_id == Hospital.hospital_id, isouter=True)
         .join(Clinic, Sociodemographics.clinic_id == Clinic.clinic_id, isouter=True)
         .join(Revision, Sociodemographics.aimedic_id == Revision.aimedic_id, isouter=True)
         )

        if n_rows is not None:
            socio_query = socio_query.limit(n_rows)

        logger.info('Reading sociodemographic and revision info ...')
        cases_df = pd.read_sql(socio_query.statement, db.session.bind)

        logger.info('Reading diagnoses codes ...')
        cases_df = get_diagnoses_codes(cases_df, db.session)
        cases_df.rename(columns={PRIMARY_DIAGNOSIS_COL: NEW_PRIMARY_DIAGNOSIS_COL}, inplace=True)

        logger.info('Reading procedure codes ...')
        cases_df = get_procedures_codes(cases_df, db.session)

    logger.success(f'Read {cases_df.shape[0]} cases')
    return cases_df
