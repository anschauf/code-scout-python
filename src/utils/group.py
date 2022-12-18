import humps
import pandas as pd
from beartype import beartype
from loguru import logger
from sqlalchemy import null

from src.models.sociodemographics import SOCIODEMOGRAPHIC_ID_COL
from src.service.aimedic_grouper import AIMEDIC_GROUPER
from src.utils.global_configs import *


@beartype
def format_for_grouper_one_case(row: pd.Series) -> pd.Series:
    """This function formats a single case for the SwissDRG grouper and is applied to the previously generated
    dataframes in function 'format_for_grouper'.
    Documentation on the grouper format: https://grouper-docs.swissdrg.org/batchgrouper2017-format.html

    @return: A series of a single revised case in the SwissDRG grouper format 2017.
    """

    sociodemographic_id = int(row[SOCIODEMOGRAPHIC_ID_COL])

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
    if primary_procedure is not None and len(primary_procedure) > 0:
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

    row[GROUPER_FORMAT_COL] = ';'.join([str(sociodemographic_id), str(age_years), str(age_days),
                                        baby_data, gender, admission_date, admission_type, discharge_date,
                                        str(discharge_type), str(duration_of_stay), str(ventilation_hours),
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
def group_revised_cases_for_db(revised_cases: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """In this function, the previously generated dataframes are formatted and grouped by the SwissDRG grouper.
    Information is given on how many cases got grouped in a logger info message.

    @return: Three dataframes with updates on revisions (grouper output), diagnoses and procedures for the DB.
    """
    # Check for unique sociodemographic IDs
    sociodemographic_ids = revised_cases[SOCIODEMOGRAPHIC_ID_COL].values.tolist()
    if len(set(sociodemographic_ids)) != (len(sociodemographic_ids)):
        raise ValueError("The provided cases don't have unique sociodemographic IDs. Make sure you pass only one revision case for one patient case.")

    logger.info(f'Grouping {revised_cases.shape[0]} cases ...')
    formatted_revised_cases = format_for_grouper(revised_cases)
    formatted_cases = formatted_revised_cases[GROUPER_FORMAT_COL].values.tolist()

    all_data = AIMEDIC_GROUPER.run_batch_grouper(formatted_cases)

    # Rename columns
    all_data.columns = [humps.decamelize(col) for col in all_data.columns]
    all_data.rename(columns={
        'id': SOCIODEMOGRAPHIC_ID_COL,
        'supplement_charges': 'supplement_charge',
        'supplement_charges_ppu': 'supplement_charge_ppu',
    }, inplace=True)

    # -------------------------------------------------------------------------
    # Select and add more columns to form the `revision_df`
    # -------------------------------------------------------------------------
    revision_df = all_data[[
        SOCIODEMOGRAPHIC_ID_COL,
        'duration_of_stay_case_type',
        'mdc',
        'mdc_partition',
        'drg',
        'drg_cost_weight',
        'effective_cost_weight',
        'pccl',
        'raw_pccl',
        'supplement_charge',
    ]].copy()

    revision_df[REVISION_ID_COL] = -1
    revision_df['reviewed'] = True
    revision_df['revised'] = True
    # revision_df[REVISION_DATE_COL] = pd.to_datetime(revision_df[REVISION_DATE_COL], format='%Y%m%d')

    # -------------------------------------------------------------------------
    # Select and add more columns to form the `diagnoses_df`
    # -------------------------------------------------------------------------
    def make_diagnosis_row(row):
        info = row['diagnoses_extended_info']
        drg_relevant_diagnoses = set(row['drg_relevant_diagnoses'])

        codes = list(info.keys())
        row['code'] = codes
        row['ccl'] = [info[code]['ccl'] for code in codes]
        row['is_primary'] = [info[code]['is_primary'] for code in codes]
        row['is_grouper_relevant'] = [code in drg_relevant_diagnoses for code in codes]
        row['global_functions'] = [tuple(info[code]['globalFunctions']) for code in codes]

        return row

    diagnoses_df = (
        all_data[[SOCIODEMOGRAPHIC_ID_COL, 'diagnoses_extended_info', 'drg_relevant_diagnoses']]
        .copy()
        .apply(make_diagnosis_row, axis=1)
        [[SOCIODEMOGRAPHIC_ID_COL, 'code', 'ccl', 'is_primary', 'is_grouper_relevant', 'global_functions']]
        .apply(pd.Series.explode)
        .sort_values(by=[SOCIODEMOGRAPHIC_ID_COL, 'is_primary', 'is_grouper_relevant', 'ccl', 'code'], ascending=[True, False, False, False, True])
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Select and add more columns to form the `procedures_df`
    # -------------------------------------------------------------------------
    def make_procedure_row(row):
        info = row['procedures_extended_info']
        drg_relevant_procedures = set(row['drg_relevant_procedures'])

        codes = list(info.keys())
        row['code'] = codes
        row['side'] = [info[code]['side'] for code in codes]
        row['date'] = [info[code]['date'] for code in codes]
        row['is_grouper_relevant'] = [code in drg_relevant_procedures for code in codes]
        row['is_primary'] = [info[code]['is_primary'] for code in codes]
        row['global_functions'] = [tuple(info[code]['globalFunctions']) for code in codes]
        row['supplement_charge'] = [info[code]['supplementCharges'] for code in codes]
        row['supplement_charge_ppu'] = [info[code]['supplementChargesPPU'] for code in codes]

        return row

    procedures_df = (
        all_data[[SOCIODEMOGRAPHIC_ID_COL, 'procedures_extended_info', 'drg_relevant_procedures']]
        .copy()
        .apply(make_procedure_row, axis=1)
        [[SOCIODEMOGRAPHIC_ID_COL, 'code', 'side', 'date', 'is_primary', 'is_grouper_relevant', 'global_functions', 'supplement_charge', 'supplement_charge_ppu']]
        .apply(pd.Series.explode)
        .sort_values(by=[SOCIODEMOGRAPHIC_ID_COL, 'is_primary', 'is_grouper_relevant', 'supplement_charge', 'supplement_charge_ppu', 'code'], ascending=[True, False, False, False, False, True])
        .reset_index(drop=True)
    )

    procedures_df[PROCEDURE_DATE_COL] = pd.to_datetime(procedures_df[PROCEDURE_DATE_COL], format='%Y%m%d', errors='coerce')
    # Replace NaT with NULL
    # REFs: https://stackoverflow.com/a/42818550, https://stackoverflow.com/a/48765738
    procedures_df[PROCEDURE_DATE_COL] = procedures_df[PROCEDURE_DATE_COL].astype(object).where(procedures_df[PROCEDURE_DATE_COL].notnull(), null())

    # Clear out empty strings
    procedures_df[PROCEDURE_SIDE_COL] = procedures_df[PROCEDURE_SIDE_COL].str.strip()

    logger.success(f'Grouped {revised_cases.shape[0]} cases into: {revision_df.shape[0]} revisions, {diagnoses_df.shape[0]} diagnoses rows, {procedures_df.shape[0]} procedure rows')
    return revision_df, diagnoses_df, procedures_df
