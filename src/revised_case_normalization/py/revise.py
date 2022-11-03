import pandas as pd
from beartype import beartype
from loguru import logger

from src.revised_case_normalization.py.global_configs import *
from src.revised_case_normalization.py.normalize import remove_leading_zeros
from src.service.bfs_cases_db_service import get_sociodemographics_for_hospital_year, \
    get_earliest_revisions_for_aimedic_ids, get_codes
from src.utils.chop_validation import split_chop_codes


@beartype
def revise(file_info: FileInfo,
           revised_cases_df: pd.DataFrame,
           *,
           validation_cols: list[str] = VALIDATION_COLS
           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Read the sociodemographics from the DB, and normalize the case ID by removing leading zeros
    cases_in_db = get_sociodemographics_for_hospital_year(file_info.hospital_name_db, int(file_info.year))
    cases_in_db[NORM_CASE_ID_COL] = cases_in_db[CASE_ID_COL].apply(remove_leading_zeros)

    # Merge cases in DB with the revised cases
    joined = pd.merge(revised_cases_df, cases_in_db,
                      how='left',
                      on=validation_cols,
                      suffixes=('', '_db'))

    # Split matched from unmatched cases
    matched_cases = joined[~joined[AIMEDIC_ID_COL].isna()].copy()
    unmatched_cases = joined[joined[AIMEDIC_ID_COL].isna()].copy()

    # Print out how many rows could not be matched
    num_unmatched = unmatched_cases.shape[0]
    if num_unmatched > 0:
        logger.warning(f'{num_unmatched} rows could not be matched, given {sorted(validation_cols)}')

    # Retrieve the codes from the DB
    original_revision_ids = get_earliest_revisions_for_aimedic_ids(matched_cases[AIMEDIC_ID_COL].values.tolist())
    original_cases = get_codes(original_revision_ids)

    # Apply the revisions to the cases from the DB
    revised_cases = apply_revisions(original_cases, matched_cases)

    # Select only the columns of interest
    revised_cases = revised_cases[[
        AIMEDIC_ID_COL,
        CASE_ID_COL,
        # codes
        NEW_PRIMARY_DIAGNOSIS_COL, SECONDARY_DIAGNOSES_COL, PRIMARY_PROCEDURE_COL, SECONDARY_PROCEDURES_COL,
        # sociodemographics
        GENDER_COL, AGE_COL, AGE_DAYS_COL, GESTATION_AGE_COL, DURATION_OF_STAY_COL, VENTILATION_HOURS_COL,
        ADMISSION_TYPE_COL, ADMISSION_DATE_COL, ADMISSION_WEIGHT_COL, DISCHARGE_TYPE_COL, DISCHARGE_DATE_COL
    ]]

    return revised_cases, unmatched_cases


@beartype
def apply_revisions(cases_df: pd.DataFrame, revisions_df: pd.DataFrame) -> pd.DataFrame:
    revised_cases = pd.merge(cases_df, revisions_df, on=AIMEDIC_ID_COL, how='left')

    # Notes:
    # - revision_id is not needed
    # - the old_pd is not needed, the new_pd is the new PD

    # Add & remove ICD codes from the list of secondary diagnoses
    def revise_diagnoses_codes(row):
        """

        @param row:
        @return:
        """
        if isinstance(row[SECONDARY_DIAGNOSES_COL], float):
            row[SECONDARY_DIAGNOSES_COL] = list()

        revised_codes = list(row[SECONDARY_DIAGNOSES_COL])

        for code_to_add in row[ADDED_ICD_CODES]:
            revised_codes.append(code_to_add)

        for code_to_remove in row[REMOVED_ICD_CODES]:
            try:
                revised_codes.remove(code_to_remove)
            except:
                print(f'{row[AIMEDIC_ID_COL]=}: {revised_codes=} - {code_to_remove=}')

        row[SECONDARY_DIAGNOSES_COL] = revised_codes
        return row

    # Delete the primary procedure if it was removed
    def revise_primary_procedure_code(row):
        """

        @param row:
        @return:
        """
        primary_chop = split_chop_codes([row[PRIMARY_PROCEDURE_COL]])[0][0]  # [0] because there is only one code, [0] to take only the CHOP code itself

        if primary_chop in row[REMOVED_CHOP_CODES]:
            row[PRIMARY_PROCEDURE_COL] = ''

        return row

    # Add & remove CHOP codes from the list of secondary procedures
    def revise_secondary_procedure_codes(row):
        """

        @param row:
        @return:
        """
        # Copy the secondary procedures into a new list
        revised_codes = list(row[SECONDARY_PROCEDURES_COL])

        # Add the new codes
        for code_to_add in row[ADDED_CHOP_CODES]:
            code_to_add_info = split_chop_codes([code_to_add])[0]  # has to be input as list and we take back the only element of the output list
            if len(code_to_add_info) == 1:  # there was only the code without side and date
                code_to_add = f'{code_to_add}::'  # append missing information for side and date

            revised_codes.append(code_to_add)

        # Split all the codes from their side and date
        revised_codes = split_chop_codes(revised_codes)

        # Get the set of codes to remove
        codes_to_remove = split_chop_codes(row[REMOVED_CHOP_CODES])  # This also takes care of revised codes in the grouper format
        codes_to_remove = {code_info[0] for code_info in codes_to_remove}

        # Discard the codes which appear in the set `codes_to_remove`
        updated_codes = list()
        for code_info in revised_codes:
            if code_info[0] not in codes_to_remove:
                updated_codes.append(code_info)

        # Join the info on each code by `:` once more, and store back into the DataFrame row
        row[SECONDARY_PROCEDURES_COL] = [':'.join(code_info) for code_info in updated_codes]
        return row

    # Apply all the revisions
    revised_cases = revised_cases.apply(revise_diagnoses_codes, axis=1)
    revised_cases = revised_cases.apply(revise_primary_procedure_code, axis=1)
    revised_cases = revised_cases.apply(revise_secondary_procedure_codes, axis=1)

    return revised_cases
