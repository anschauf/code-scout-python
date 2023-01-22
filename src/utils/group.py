import pandas as pd
from beartype import beartype
from loguru import logger
from tqdm import tqdm

from src.models.sociodemographics import SOCIODEMOGRAPHIC_ID_COL
from src.service.aimedic_grouper import group_batch_group_cases
from src.utils.global_configs import *

tqdm.pandas()


@beartype
def format_for_grouper(revised_cases: pd.DataFrame, *, with_sociodemographic_id: bool = True) -> pd.DataFrame:

    """Given the previously generated dataframes, this function formats the datasets for the SwissDRG grouper.

    @return: A string of the revised cases in the SwissDRG grouper format 2017
    """

    @beartype
    def format_for_grouper_one_case(row: pd.Series) -> pd.Series:
        """This function formats a single case for the SwissDRG grouper and is applied to the previously generated
         dataframes in function 'format_for_grouper'.
        Documentation on the grouper format: https://grouper-docs.swissdrg.org/batchgrouper2017-format.html

           @return: A series of a single revised case in the SwissDRG grouper format 2017.
        """

        sociodemographic_id = int(row[SOCIODEMOGRAPHIC_ID_COL])
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

        # Concatenate all the procedure codes
        all_procedure_codes = list()
        primary_procedure = str(row[PRIMARY_PROCEDURE_COL])
        if primary_procedure is not None and len(primary_procedure) > 0 and 'nan' not in primary_procedure:
            all_procedure_codes.append(primary_procedure)

        secondary_procedures = row[SECONDARY_PROCEDURES_COL]
        if secondary_procedures is not None and len(secondary_procedures) > 0:
            all_procedure_codes.extend(secondary_procedures)

        procedures = '|'.join(all_procedure_codes)

        # Concatenate all the diagnosis codes
        all_diagnosis_codes = list()
        primary_diagnosis = str(row[NEW_PRIMARY_DIAGNOSIS_COL])
        if primary_diagnosis is not None and len(primary_diagnosis) > 0:
            all_diagnosis_codes.append(primary_diagnosis)

        secondary_diagnoses = row[SECONDARY_DIAGNOSES_COL]
        if secondary_diagnoses is not None and len(secondary_diagnoses) > 0:
            all_diagnosis_codes.extend(secondary_diagnoses)

        diagnoses = '|'.join(all_diagnosis_codes)

        medications = row[MEDICATIONS_COL]
        if len(medications) > 0:
            medications = medications.replace(';', ':')

        fields = list()
        if with_sociodemographic_id:
            fields.append(str(sociodemographic_id))

        fields.extend([str(case_id), str(age_years), str(age_days),
                       baby_data, gender, admission_date, admission_type, discharge_date,
                       str(discharge_type), str(duration_of_stay), str(ventilation_hours),
                       diagnoses, procedures, medications])

        fields_str = ';'.join(fields)

        if 'nan' in fields_str:
            raise ValueError(f'There was a nan in\n{fields_str}')

        row[GROUPER_FORMAT_COL] = fields_str
        return row

    revised_cases_formatted = revised_cases.progress_apply(format_for_grouper_one_case, axis=1)
    return revised_cases_formatted


@beartype
def group(revised_cases: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """In this function, the previously generated dataframes are formatted and grouped by the SwissDRG grouper.
    Information is given on how many cases got grouped in a logger info message.

    @return: Three dataframes with updates on revisions (grouper output), diagnoses and procedures for the DB.
    """
    logger.info(f'Grouping {revised_cases.shape[0]} cases ...')
    formatted_revised_cases = format_for_grouper(revised_cases)
    formatted_cases = formatted_revised_cases[GROUPER_FORMAT_COL].values.tolist()
    revision_df, diagnoses_df, procedures_df = group_batch_group_cases(formatted_cases)

    logger.success(f'Grouped {revised_cases.shape[0]} cases into: {revision_df.shape[0]} revisions, {diagnoses_df.shape[0]} diagnoses rows, {procedures_df.shape[0]} procedure rows')
    return revision_df, diagnoses_df, procedures_df