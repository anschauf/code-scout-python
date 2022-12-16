import pandas as pd
from beartype import beartype
from loguru import logger

from src.models.sociodemographics import SOCIODEMOGRAPHIC_ID_COL
from src.service.bfs_cases_db_service import get_codes, get_original_revision_id_for_sociodemographic_ids, \
    get_sociodemographics_for_hospital_year
from src.service.database import Database
from src.utils.global_configs import *
from src.utils.normalize import remove_leading_zeros
from src.utils.revised_case_files_info import FileInfo


@beartype
def revise(file_info: FileInfo,
           revised_cases_df: pd.DataFrame,
           *,
           validation_cols: list[str] = VALIDATION_COLS
           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lookup the revised cases in our DB, and return them with additional infos, such as the primary keys which are
    used to insert these new cases and link them to the existing cases to which they refer.

    @param file_info: An instance of FileInfo, which contains the name and path of filename, hospital name and year and
        the sheet name to analyze.
    @param revised_cases_df: A dataframe of revised cases after normalization..
    @param validation_cols: List of column's names used to validate the revised cases with database cases.

    @return: Two pandas DataFrame: one contains matched revised cases with DB, one contains unmatched revises cases.
    """
    with Database() as db:
        # Read the sociodemographics from the DB, and normalize the case ID by removing leading zeros
        cases_in_db = get_sociodemographics_for_hospital_year(file_info.hospital_name_db, int(file_info.year), db.session)
        cases_in_db[NORM_CASE_ID_COL] = cases_in_db[CASE_ID_COL].apply(remove_leading_zeros)

        # Merge cases in DB with the revised cases
        joined = pd.merge(revised_cases_df, cases_in_db,
                          how='left', on=validation_cols,
                          suffixes=('', '_db'))

        unique_joined = joined.drop_duplicates(validation_cols)
        if unique_joined.shape[0] != joined.shape[0]:
            raise ValueError(f'Found duplicates based on {validation_cols}. Make the criteria stricter not to match multiple cases per revised case')

        # Split matched from unmatched cases
        matched_cases = joined[~joined[AIMEDIC_ID_COL].isna()].copy()
        unmatched_cases = joined[joined[AIMEDIC_ID_COL].isna()].copy()

        num_unmatched = unmatched_cases.shape[0]
        if num_unmatched > 0:
            logger.warning(f'{num_unmatched} cases could not be matched, given {sorted(validation_cols)}:\n{unmatched_cases[[PATIENT_ID_COL, CASE_ID_COL]]}')

        # Retrieve the codes from the DB
        original_revision_ids = get_original_revision_id_for_sociodemographic_ids(matched_cases[SOCIODEMOGRAPHIC_ID_COL].astype(int).values.tolist(), db.session)
        original_cases = get_codes(original_revision_ids, db.session)

    # Apply the revisions to the cases from the DB
    revised_cases = apply_revisions(original_cases, matched_cases)

    # Select only the columns of interest
    revised_cases = revised_cases[[
        # IDs
        SOCIODEMOGRAPHIC_ID_COL, CASE_ID_COL,
        # codes
        NEW_PRIMARY_DIAGNOSIS_COL, SECONDARY_DIAGNOSES_COL, PRIMARY_PROCEDURE_COL, SECONDARY_PROCEDURES_COL,
        # sociodemographics
        GENDER_COL, AGE_COL, AGE_DAYS_COL, GESTATION_AGE_COL, DURATION_OF_STAY_COL, VENTILATION_HOURS_COL,
        ADMISSION_TYPE_COL, ADMISSION_DATE_COL, ADMISSION_WEIGHT_COL, DISCHARGE_TYPE_COL, DISCHARGE_DATE_COL,
        MEDICATIONS_COL, REVISION_DATE_COL
    ]]

    # Format columns to integer before calling the group function
    for col in (AGE_DAYS_COL, GESTATION_AGE_COL, VENTILATION_HOURS_COL, ADMISSION_WEIGHT_COL):
        revised_cases[col] = revised_cases[col].astype(int)

    return revised_cases, unmatched_cases


def __revise_diagnoses_codes(row):
    """
    Update diagnoses codes for a revised case.
    Add & remove ICD codes from the list of secondary diagnoses

    Notes:
    - revision_id is not needed
    - the old_pd is not needed, the new_pd is the new PD
    """
    if isinstance(row[SECONDARY_DIAGNOSES_COL], float):
        row[SECONDARY_DIAGNOSES_COL] = list()

    revised_codes = list(row[SECONDARY_DIAGNOSES_COL])

    for code_to_add in row[ADDED_ICD_CODES]:
        revised_codes.append(code_to_add)

    for code_to_remove in row[REMOVED_ICD_CODES]:
        try:
            revised_codes.remove(code_to_remove)
        except Exception as e:
            print(f'{row[AIMEDIC_ID_COL]=}: {revised_codes=} - {code_to_remove=}: {e}')

    row[SECONDARY_DIAGNOSES_COL] = revised_codes
    return row


def __revise_primary_procedure_code(row):
    """
    Update primary procedure for a revised case.
    Delete the primary procedure if it was removed
    """
    primary_chop = row[PRIMARY_PROCEDURE_COL]

    if primary_chop in row[REMOVED_CHOP_CODES]:
        row[PRIMARY_PROCEDURE_COL] = ''

    return row


def __revise_secondary_procedure_codes(row):
    """
    Update secondary procedure for a revised case.
    Add & remove CHOP codes from the list of secondary procedures
    """
    if isinstance(row[SECONDARY_PROCEDURES_COL], float):
        row[SECONDARY_PROCEDURES_COL] = list()

    # Copy the secondary procedures into a new list
    revised_codes = list(row[SECONDARY_PROCEDURES_COL])

    for code_to_add in row[ADDED_CHOP_CODES]:
        revised_codes.append(code_to_add)

    if len(revised_codes) > 0:
        for code_to_remove in row[REMOVED_CHOP_CODES]:
            try:
                revised_codes.remove(code_to_remove)
            except ValueError:
                logger.error(f'{row[CASE_ID_COL]=}: Cannot remove [{code_to_remove}] from [{revised_codes}]')

    row[SECONDARY_PROCEDURES_COL] = revised_codes
    return row


@beartype
def apply_revisions(cases_df: pd.DataFrame, revisions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply revision to matched cases from DB and update diagonoses and procedure codes based on revision.
    @param cases_df: A pandas Dataframe contains codes from DB before revision.
    @param revisions_df: A pandas DataFrame contains codes to be revised, i.e. added and removed diagonoses and procedure codes
    @return: a pandas DataFrame: contains revised cases after updating diagonoses and procedure codes
    """
    revised_cases = pd.merge(cases_df, revisions_df, on=SOCIODEMOGRAPHIC_ID_COL, how='left')

    # Apply all the revisions
    revised_cases = revised_cases.apply(__revise_diagnoses_codes, axis=1)
    revised_cases = revised_cases.apply(__revise_primary_procedure_code, axis=1)
    revised_cases = revised_cases.apply(__revise_secondary_procedure_codes, axis=1)

    return revised_cases

