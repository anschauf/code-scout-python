import pandas as pd
import numpy as np
from src.revised_case_normalization.py.global_configs import *
from src.service.bfs_cases_db_service import get_earliest_revisions_for_aimedic_ids, get_codes, apply_revisions
from loguru import logger


def format_for_grouper(df_joined: pd.DataFrame) -> pd.DataFrame:
    """ Given the previously generated dataframes, this function formats the datasets for the SwissDRG grouper.

        @return: A string of the revised cases in the SwissDRG Batchgrouper format 2017
        """
    joined = df_joined.copy()

    # Select cases in which the aimedic_id is not an NA
    joined = joined[joined['aimedic_id'].notna()]
    joined = joined.replace(np.nan, "")
    joined['aimedic_id'] = joined['aimedic_id'].astype(int)

    # set type of age_days, admission_weight and gestation_age to integer (to avoid float format)
    joined['age_days'] = joined['age_days'].astype(int)
    joined['admission_weight'] = joined['admission_weight'].astype(int)
    joined['gestation_age'] = joined['gestation_age'].astype(int)

    # Formatting baby data

    joined["baby_data"] = joined['admission_weight'].map(str) + "|" + joined['gestation_age'].map(str)
    joined["baby_data"] = joined["baby_data"].replace("0|0", "")

    # Extract grouper relevant columns
    joined = joined[
        GROUPER_INPUT_BFS + [NEW_PRIMARY_DIAGNOSIS_COL, ADDED_ICD_CODES, REMOVED_ICD_CODES, ADDED_CHOP_CODES,
                             REMOVED_CHOP_CODES]]

    # Format admission_date and discharge_date the (SwissDRG Batchgrouper Format 2017 (YYYYMMDD)
    joined['admission_date'] = joined['admission_date'].astype(str).str.replace("-", "")
    joined['discharge_date'] = joined['discharge_date'].astype(str).str.replace("-", "")

    original_revision_ids = get_earliest_revisions_for_aimedic_ids(joined[AIMEDIC_ID_COL].values.tolist())
    original_cases = get_codes(original_revision_ids)

    revised_cases = apply_revisions(original_cases, joined)

    # Formatting primary_procedure and secondary_procedures to fit SwissDRG Batchgrouper Format 2017
    # NOTE: Sideness and procedure date are not taken into account

    # Formatting primary procedure column (condition: add "::" only if the chop code is available)
    revised_cases["primary_procedure"] = [procedure + "::" for procedure in revised_cases["primary_procedure"]]
    revised_cases['primary_procedure'] = revised_cases.primary_procedure.apply(lambda case: case + "::" if len(case) <= 6 else case)

    # Formatting secondary procedure column (condition: add "::" only if the chop code is available)
    revised_cases["secondary_procedures"] = revised_cases['secondary_procedures'].map(str).str.strip("[]")
    revised_cases['secondary_procedures'] = revised_cases.secondary_procedures.apply(lambda case: case + "::" if len(case) <= 6 else case)
    revised_cases["secondary_procedures"] = revised_cases["secondary_procedures"].str.replace("'", "").str.replace(",","::|").str.replace(" ", "")

    # Formatting grouper procedures column
    revised_cases["grouper_procedures"] = revised_cases['primary_procedure'].map(str) + "|" + revised_cases[
        'secondary_procedures'].map(str)

    revised_cases["grouper_procedures"] = revised_cases["grouper_procedures"].str.rstrip("::|")

    # Formatting primary_diagnosis and secondary_diagnosis to fit SwissDRG Batchgrouper Format 2017

    revised_cases["secondary_diagnoses"] = revised_cases['secondary_diagnoses'].map(str).str.strip("[]")
    revised_cases["secondary_diagnoses"] = revised_cases["secondary_diagnoses"].str.replace("'", "").str.replace(",","|").str.replace(" ", "")

    revised_cases["grouper_diagnoses"] = revised_cases['primary_diagnosis'].map(str) + "|" + revised_cases[
        "secondary_diagnoses"].map(str)

    # Extract and reorder relevant columns from BFS DB data
    joined_grouper = joined[GROUPER_INPUT_BFS]

    # Extract and reorder relevant columns from revised cases
    revised_cases_grouper = revised_cases[GROUPER_INPUT_REVISED_CASES]

    # Join revised diagnoses and procedures to cases in db:

    grouper_input_data = pd.merge(joined_grouper, revised_cases_grouper, how='inner', on='aimedic_id',
                                  suffixes=('', '_db'))

    # Added empty medication column (Placeholder!!)

    grouper_input_data["medication"] = ""

    # Format to string file for grouper with necessary ";" delimiter

    grouper_input_data_string = grouper_input_data.astype(str).apply(';'.join, axis=1)

    # Generate Log message
    n_grouper = grouper_input_data_string.shape[0]
    logger.info(f'Formatted {n_grouper} cases')

    return grouper_input_data_string
