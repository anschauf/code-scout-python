from decimal import *

import pandas as pd
from beartype import beartype
from loguru import logger
from pandas import DataFrame
from sqlalchemy import tuple_
from sqlalchemy.orm.session import Session
from sqlalchemy.sql.elements import Null

from src.models.clinic import Clinic
from src.models.diagnosis import Diagnosis
from src.models.duration_of_stay import DurationOfStay
from src.models.hospital import Hospital
from src.models.procedure import Procedure
from src.models.revision import Revision
from src.models.sociodemographics import Sociodemographics, SOCIODEMOGRAPHIC_ID_COL
from src.utils.global_configs import *


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
def get_sociodemographics_by_sociodemographics_ids(sociodemographics_ids: list, session: Session) -> DataFrame:
    """
     Get records from case_data.Sociodemographics table using sociodemographics_ids
     @param sociodemographics_ids: a list of sociodemographics_ids
     @return: a Dataframe from sociodemographic table
     """
    query = (session
             .query(Sociodemographics)
             .filter(Sociodemographics.sociodemographic_id.in_(sociodemographics_ids)))

    return pd.read_sql(query.statement, session.bind)

@beartype
def get_sociodemographics_by_case_id(case_ids: list, session: Session) -> DataFrame:
    """
    Get records from case_data.Sociodemographics table using case_id.
    @param case_ids: a list of case_ids
    @param session: DB session
    @return: a Dataframe from sociodemographic table
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
def get_clinics(session: Session) -> pd.DataFrame:
    """
    Get all records from dimension.clinic as a pandas dataframe
    """
    query = session.query(Clinic)
    return pd.read_sql(query.statement, session.bind)


@beartype
def get_duration_of_stay_df(session: Session) -> pd.DataFrame:
    """
    Get all records from dimension.duration_of_stay table
    """
    query = session.query(DurationOfStay)
    return pd.read_sql(query.statement, session.bind)


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
        .join(Hospital, Sociodemographics.hospital_id == Hospital.hospital_id)
        .filter(Hospital.hospital_name == hospital_name)
        .filter(Sociodemographics.discharge_year == year)
        .filter(Sociodemographics.case_id != '')
    )

    df = pd.read_sql(query_sociodemo.statement, session.bind)

    num_cases_in_db = df.shape[0]
    if num_cases_in_db == 0:
        raise ValueError(f"There is no data for the hospital '{hospital_name}' in {year}")

    return df


@beartype
def get_all_revised_cases(session: Session) -> pd.DataFrame:
    """
    Get all revised cases from revision table
    @return: a dataframe with all revised cases from revision table
    """
    query_revised_case = (
        session.query(Revision).
        filter(Revision.revised.is_(True)))
    df = pd.read_sql(query_revised_case.statement, session.bind)

    return df


@beartype
def get_original_revision_id_for_sociodemographic_ids(sociodemographic_ids: list[int],
                                                      session: Session) -> pd.DataFrame:
    """
    Get the original revisions ids of sociodemographic_ids
    @param session: active DB session
    @param sociodemographic_ids:
    @return: a Dataframe containing sociodemographic ids, revision ids
    """

    query_revisions = (
        session
        .query(Revision)
        .with_entities(Revision.revision_id, Revision.sociodemographic_id)
        .filter(Revision.sociodemographic_id.in_(sociodemographic_ids))
        .filter(Revision.revised.is_(False))
        .filter(Revision.reviewed.is_(False))
    )

    df = pd.read_sql(query_revisions.statement, session.bind)

    return df


def get_grouped_revisions_for_sociodemographic_ids(sociodemographic_ids: list[str], session: Session) -> DataFrame:
    """
    Get the grouped revisions for sociodemographic_ids.
    @param sociodemographic_ids: a list of sociodemographic_ids
    @param session: The DB session.
    """

    query = (
        session
        .query(Revision)
        .filter(Revision.sociodemographic_id.in_(sociodemographic_ids))
    )

    df = pd.read_sql(query.statement, session.bind)
    return df.groupby(SOCIODEMOGRAPHIC_ID_COL, as_index=False).agg(lambda x: list(x))

@beartype
def get_all_reviewed_cases(session: Session) -> pd.DataFrame:
    """
    Get all reviewed cases (i.e. reviewed; True, revised = False)
    @param session:
    @return: pandas dataframe
    """

    query_reviewed_case = (
        session.query(Revision)
        .filter(Revision.reviewed.is_(True))
        .filter(Revision.revised.is_(False)))
    df = pd.read_sql(query_reviewed_case.statement, session.bind)

    return df


def get_revision_for_revision_ids(revision_ids: list[int],
                                  session: Session) -> pd.DataFrame:
    """
    Get revisions for revision_ids
    @param session: active DB session
    @param revision_ids:
    @return: a Dataframe from all columns from revision table
    """

    query_revisions = (
        session
        .query(Revision)
        .filter(Revision.revision_id.in_(revision_ids))
    )

    df = pd.read_sql(query_revisions.statement, session.bind)

    return df


@beartype
def get_diagnoses_codes(df_revision_ids: pd.DataFrame, session: Session) -> pd.DataFrame:
    """
     Retrieve primary and secondary diagnoses of the revised cases from the DB.
     @param session: active DB session
     @param df_revision_ids: a Dataframe with sociodemographic_id and revision_id
     @return: a Dataframe containing revision ids, primary and secondary diagnoses
     """

    all_revision_ids = set(df_revision_ids['revision_id'].values.tolist())

    query_diagnoses = (
        session
        .query(Diagnosis)
        .with_entities(Diagnosis.sociodemographic_id, Diagnosis.revision_id, Diagnosis.code, Diagnosis.is_primary)
        .filter(Diagnosis.revision_id.in_(all_revision_ids))
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
    codes_df[SECONDARY_DIAGNOSES_COL] = codes_df[SECONDARY_DIAGNOSES_COL].apply(
        lambda x: x if isinstance(x, list) else [])

    return codes_df

@beartype
def get_diagnoses_codes_from_revision_id(revision_ids: list[int], session: Session) -> pd.DataFrame:

    query_diagnoses = (
        session
        .query(Diagnosis)
        .with_entities(Diagnosis.sociodemographic_id, Diagnosis.revision_id, Diagnosis.code)
        .filter(Diagnosis.revision_id.in_(revision_ids))
    )

    df = pd.read_sql(query_diagnoses.statement, session.bind)
    return df


@beartype
def get_procedures_codes_from_revision_id(revision_ids: list[int], session: Session) -> pd.DataFrame:

    query_procedures = (
        session
        .query(Procedure)
        .with_entities(Procedure.sociodemographic_id, Procedure.revision_id, Procedure.code)
        .filter(Procedure.revision_id.in_(revision_ids))
    )

    df = pd.read_sql(query_procedures.statement, session.bind)
    return df

@beartype
def get_procedures_codes(df_revision_ids: pd.DataFrame, session: Session) -> pd.DataFrame:
    """
     Retrieve primary and secondary procedures of the revised cases from the DB.
     @param session: active DB session
     @param df_revision_ids: a Dataframe with aimedic_id and revision_id
     @return: a dataframe containing revision ids, primary and secondary diagnoses
     """

    all_revision_ids = set(df_revision_ids['revision_id'].values.tolist())

    query_procedures = (
        session
        .query(Procedure)
        .with_entities(Procedure.sociodemographic_id, Procedure.revision_id, Procedure.code, Procedure.side,
                       Procedure.date, Procedure.is_primary)
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

    secondary_procedures = secondary_procedures.groupby(REVISION_ID_COL, group_keys=True)[
        SECONDARY_PROCEDURES_COL].apply(list)

    codes_df = (df_revision_ids
                .merge(primary_procedure, on=REVISION_ID_COL, how='left')
                .merge(secondary_procedures, on=REVISION_ID_COL, how='left'))

    # Fill NaNs
    codes_df[PRIMARY_PROCEDURE_COL] = codes_df[PRIMARY_PROCEDURE_COL].fillna('')
    codes_df[SECONDARY_PROCEDURES_COL] = codes_df[SECONDARY_PROCEDURES_COL].apply(
        lambda x: x if isinstance(x, list) else [])

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

    # Drop the sociodemographic_id column to avoid adding it with a suffix and having to remove it later
    codes_df = (df_revision_ids
                .merge(diagnoses_df.drop(columns=SOCIODEMOGRAPHIC_ID_COL), on='revision_id', how='left')
                .merge(procedures_df.drop(columns=SOCIODEMOGRAPHIC_ID_COL), on='revision_id', how='left'))

    return codes_df


@beartype
def insert_revised_cases_into_revisions(revised_case_revision_df: pd.DataFrame, session: Session) -> dict:
    """
    Insert revised cases into the table coding_revision.revisions
    @param session: active DB session
    @param revised_case_revision_df: a Dataframe of revised cases after grouping
    @return: a dictionary with sociodemographic_ids as keys and revision_ids as values created after insert into the DB
    """
    logger.info(f"Trying to insert {revised_case_revision_df.shape[0]} cases into the 'Revisions' table ...")
    revision_list = revised_case_revision_df.to_dict(orient='records')

    values_to_insert = list()

    for revision in revision_list:
        values_to_insert.append({
            "sociodemographic_id": int(revision[SOCIODEMOGRAPHIC_ID_COL]),
            "drg": str(revision[DRG_COL]),
            "drg_cost_weight": float(revision[DRG_COST_WEIGHT_COL]),
            "effective_cost_weight": float(revision[EFFECTIVE_COST_WEIGHT_COL]),
            "pccl": int(revision[PCCL_COL]),
            'dos_id': int(revision['dos_id']),
            'mdc': str(revision['mdc']),
            'mdc_partition': str(revision['mdc_partition']),
            'raw_pccl': float(revision['raw_pccl']),
            'supplement_charge': Decimal(revision['supplement_charge']),
            'supplement_charge_ppu': Decimal(revision['supplement_charge_ppu']),
            'reviewed': bool(revision['reviewed']),
            'revised': bool(revision['revised']),
            "revision_date": str(revision[REVISION_DATE_COL])
        })

    values_info = [(values_dict[SOCIODEMOGRAPHIC_ID_COL], values_dict[REVISION_DATE_COL]) for values_dict in
                   values_to_insert]

    num_rows_before = session.query(Revision).count()
    delete_statement = (Revision.__table__
                        .delete()
                        .where(tuple_(Revision.sociodemographic_id, Revision.revision_date).in_(values_info)))
    session.execute(delete_statement)
    session.commit()

    num_rows_after = session.query(Revision).count()
    if num_rows_after != num_rows_before:
        logger.info(
            f"Deleted {num_rows_before - num_rows_after} rows from the 'Revisions' table, which is about to be updated")

    insert_statement = (Revision.__table__
                        .insert()
                        .values(values_to_insert)
                        .returning(Revision.sociodemographic_id, Revision.revision_id))

    result = session.execute(insert_statement).fetchall()
    session.commit()

    sociodemographic_id_with_revision_id = {sociodemographic_id: revision_id for sociodemographic_id, revision_id in
                                            result}
    logger.success(f"Inserted {len(result)} cases into the 'Revisions' table")
    return sociodemographic_id_with_revision_id


@beartype
def insert_revised_cases_into_diagnoses(revised_case_diagnoses: pd.DataFrame,
                                        sociodemographic_id_with_revision_id: dict, session: Session):
    """
    Insert revised cases into the table coding_revision.diagnoses
    @param session: active DB session
    @param revised_case_diagnoses: a Dataframe of revised cases for diagnoses after grouping
    @param sociodemographic_id_with_revision_id: a dictionary with sociodemographic_ids as keys and revision_ids as values which are created
        after insert into the DB
    """
    logger.info(f"Trying to insert {revised_case_diagnoses.shape[0]} rows into the 'Diagnoses' table ...")

    diagnosis_list = revised_case_diagnoses.to_dict(orient='records')

    values_to_insert = list()

    for diagnoses in diagnosis_list:
        sociodemographic_id = int(diagnoses[SOCIODEMOGRAPHIC_ID_COL])

        values_to_insert.append({
            "sociodemographic_id": sociodemographic_id,
            "revision_id": int(sociodemographic_id_with_revision_id[sociodemographic_id]),
            "code": str(diagnoses[CODE_COL]),
            "ccl": int(diagnoses[CCL_COL]),
            "is_primary": bool(diagnoses[IS_PRIMARY_COL]),
            "is_grouper_relevant": bool(diagnoses[IS_GROUPER_RELEVANT_COL]),
            'global_functions': str(diagnoses['global_functions'])
        })

    insert_statement = (Diagnosis.__table__
                        .insert()
                        .values(values_to_insert))

    session.execute(insert_statement)
    session.commit()

    logger.success(f"Inserted {len(values_to_insert)} rows into the 'Diagnoses' table")


@beartype
def insert_revised_cases_into_procedures(revised_case_procedures: pd.DataFrame,
                                         sociodemographic_id_with_revision_id: dict, session: Session):
    """Insert revised cases into table coding_revision.procedures.

    @param revised_case_procedures: a Dataframe of revised case for procedures after grouping.
    @param sociodemographic_id_with_revision_id: a dictionary with sociodemographic_id as keys and revision_id as values which are created
    after insert into the DB.
    """
    logger.info(f"Trying to insert {revised_case_procedures.shape[0]} rows into the 'Procedures' table ...")

    procedure_list = revised_case_procedures.to_dict(orient='records')

    values_to_insert = list()

    for procedure in procedure_list:
        sociodemographic_id = int(procedure[SOCIODEMOGRAPHIC_ID_COL])

        # Get the procedure date as None or as a string
        procedure_date = procedure[PROCEDURE_DATE_COL]
        if procedure_date is None or isinstance(procedure_date, Null):
            procedure_date = None
        else:
            procedure_date = str(procedure_date)

        values_to_insert.append({
            "sociodemographic_id": sociodemographic_id,
            "revision_id": int(sociodemographic_id_with_revision_id[sociodemographic_id]),
            "code": str(procedure[CODE_COL]),
            "side": str(procedure[PROCEDURE_SIDE_COL]),
            "date": procedure_date,
            "is_primary": bool(procedure[IS_PRIMARY_COL]),
            "is_grouper_relevant": bool(procedure[IS_GROUPER_RELEVANT_COL]),
            "global_functions": str(procedure['global_functions']),
            "supplement_charge": Decimal(procedure['supplement_charge']),
            "supplement_charge_ppu": Decimal(procedure['supplement_charge_ppu']),

        })

    insert_statement = (Procedure.__table__
                        .insert()
                        .values(values_to_insert))

    session.execute(insert_statement)
    session.commit()

    logger.success(f"Inserted {len(values_to_insert)} rows into the 'Procedures' table")


def get_revised_case_with_codes_after_revision(session: Session) -> pd.DataFrame:
    """
    Get all revised cases with diagnoses and procedures
    @param session:
    @return: a dataframe
    """
    revised_case = get_all_revised_cases(session)
    df_diagnoses = get_diagnoses_codes(revised_case, session)
    df_procedures = get_procedures_codes(revised_case, session)

    #  reset index as revision_id
    revised_case.set_index(REVISION_ID_COL, inplace=True)
    df_diagnoses.set_index(REVISION_ID_COL, inplace=True)
    df_procedures.set_index(REVISION_ID_COL, inplace=True)

    #  merge all data using revision_id
    revised_cases_df = pd.concat([revised_case, df_diagnoses[[PRIMARY_DIAGNOSIS_COL, SECONDARY_DIAGNOSES_COL]],
                                  df_procedures[[PRIMARY_PROCEDURE_COL, SECONDARY_PROCEDURES_COL]]], axis=1).reset_index()
    revised_cases_df.rename(columns={'old_pd': 'pd'}, inplace=True)

    return revised_cases_df


def get_revised_case_with_codes_before_revision(session: Session) -> pd.DataFrame:
    """
    Get original cases with diagnoses and procedures for all revised cases
    @param session:
    @return: a dataframe
    """
    revised_cases = get_all_revised_cases(session)
    revised_case_sociodemographic_ids = revised_cases[SOCIODEMOGRAPHIC_ID_COL].values.tolist()
    revised_case_all_df = get_original_revision_id_for_sociodemographic_ids(revised_case_sociodemographic_ids, session)
    revision_ids = revised_case_all_df[REVISION_ID_COL].values.tolist()

    revised_case_orig = get_revision_for_revision_ids(revision_ids, session)
    df_diagnoses = get_diagnoses_codes(revised_case_orig, session)
    df_procedures = get_procedures_codes(revised_case_orig, session)

    #  reset index as revision_id
    revised_case_orig.set_index(REVISION_ID_COL, inplace=True)
    df_diagnoses.set_index(REVISION_ID_COL, inplace=True)
    df_procedures.set_index(REVISION_ID_COL, inplace=True)

    #  merge all data using revision_id
    revised_cases_before_revision = pd.concat(
        [revised_case_orig, df_diagnoses[[PRIMARY_DIAGNOSIS_COL, SECONDARY_DIAGNOSES_COL]],
         df_procedures[[PRIMARY_PROCEDURE_COL, SECONDARY_PROCEDURES_COL]]], axis=1).reset_index()
    revised_cases_before_revision.rename(columns={'old_pd': 'pd'}, inplace=True)
    return revised_cases_before_revision

def get_all_diagonosis(session: Session) -> pd.DataFrame:
    """
    Get all diagnoses from database
    @param session:
    @return: a dataframe with code as a list for each case
    """

    revised_cases = get_all_revised_cases(session)
    revised_case_sociodemographic_ids = revised_cases[SOCIODEMOGRAPHIC_ID_COL].values.tolist()
    original_revision_ids_revised_cases = get_original_revision_id_for_sociodemographic_ids(revised_case_sociodemographic_ids, session)[REVISION_ID_COL].tolist()

    query_diagnoses = (
        session
        .query(Diagnosis)
        .with_entities(Diagnosis.diagnoses_pk, Diagnosis.sociodemographic_id,Diagnosis.revision_id, Diagnosis.code, Diagnosis.is_primary, Diagnosis.ccl)
        .filter(Diagnosis.revision_id.notin_(original_revision_ids_revised_cases))
        # .group_by(Diagnosis.revision_id)
       )

    all_diagonosis_df = pd.read_sql(query_diagnoses.statement, session.bind)
    # all_diagonosis_df.groupby()
    # delete original cases which are revised

    return all_diagonosis_df