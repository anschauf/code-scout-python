import pandas as pd
from beartype import beartype
from loguru import logger

from src.revised_case_normalization.notebook_functions.global_configs import *
from src.service.aimedic_grouper import group_batch_group_cases

@beartype
def format_for_grouper_one_case(row: pd.Series) -> pd.Series:

    """This function formats a single case for the SwissDRG grouper and is applied to the previously generated dataframes
       in function 'format_for_grouper'.
       Documentation on the grouper format: https://grouper-docs.swissdrg.org/batchgrouper2017-format.html

       @return: A series of a single revised case in the SwissDRG grouper format 2017.
       """

    aimedic_id = int(row[AIMEDIC_ID_COL])
    case_id = int(row[CASE_ID_COL])

    age_years = int(row[AGE_COL])
    age_days = int(row[AGE_DAYS_COL])
    admission_weight = int(row[ADMISSION_WEIGHT_COL])
    gestation_age = int(row[GESTATION_AGE_COL])

    duration_of_stay = int(row[DURATION_OF_STAY_COL])
    ventilation_hours = int(row[VENTILATION_HOURS_COL])

    if admission_weight == 0 and gestation_age == 0:
        baby_data = ''
    else:
        baby_data = f'{admission_weight}|{gestation_age}'

    gender = row[GENDER_COL]

    admission_date = str(row[ADMISSION_DATE_COL]).replace("-", "")
    admission_type = row[ADMISSION_TYPE_COL]
    discharge_date = str(row[DISCHARGE_DATE_COL]).replace("-", "")
    discharge_type = row[DISCHARGE_TYPE_COL]

    primary_procedure = row[PRIMARY_PROCEDURE_COL]
    secondary_procedures = '|'.join(row[SECONDARY_PROCEDURES_COL])

    # check if primary procedure is not defined or not an empty str
    if not primary_procedure or len(primary_procedure) == 0:
        procedures = f'{secondary_procedures}'
    else:
        procedures = f'{primary_procedure}|{secondary_procedures}'

    primary_diagnosis = str(row[NEW_PRIMARY_DIAGNOSIS_COL])
    secondary_diagnoses = '|'.join(row[SECONDARY_DIAGNOSES_COL])
    diagnoses = f'{primary_diagnosis}|{secondary_diagnoses}'

    medications = ''

    row[GROUPER_FORMAT_COL] = ';'.join([str(aimedic_id), str(case_id), str(age_years), str(age_days), baby_data, gender,
                                        admission_date, admission_type, discharge_date, discharge_type,
                                        str(duration_of_stay), str(ventilation_hours),
                                        diagnoses, procedures, medications])
    return row


@beartype
def format_for_grouper(revised_cases: pd.DataFrame) -> pd.DataFrame:

    """Given the previously generated dataframes, this function formats the datasets for the SwissDRG grouper.

    @return: A string of the revised cases in the SwissDRG grouper format 2017
    """
    revised_cases_formatted = revised_cases.apply(format_for_grouper_one_case, axis=1)
    return revised_cases_formatted


@beartype
def group(revised_cases: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    """In this function, the previously generated dataframes are formatted and grouped by the SwissDRG grouper.
        Information is given on how many cases got grouped in a logger info message.

       @return: Three dataframes with updates on revisions (grouper output), diagnoses and procedures for the DB.
       """

    logger.info(f'Grouping {revised_cases.shape[0]} cases ...')
    formatted_revised_cases = format_for_grouper(revised_cases)
    revision_df, diagnoses_df, procedures_df = group_batch_group_cases(
        formatted_revised_cases[GROUPER_FORMAT_COL].tolist())

    logger.success(
        f'Grouped {revised_cases.shape[0]} cases into: {revision_df.shape[0]} revisions, {diagnoses_df.shape[0]} diagnoses rows, {procedures_df.shape[0]} procedure rows')
    return revision_df, diagnoses_df, procedures_df