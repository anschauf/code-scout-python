import json
import os.path
import subprocess

import pandas as pd
from beartype import beartype
from loguru import logger
from sqlalchemy.sql import null

from src import ROOT_DIR
from src.revised_case_normalization.notebook_functions.global_configs import AIMEDIC_ID_COL, DRG_COST_WEIGHT_COL, \
    EFFECTIVE_COST_WEIGHT_COL, REVISION_DATE_COL, IS_GROUPER_RELEVANT_COL, IS_PRIMARY_COL, CCL_COL, CODE_COL, \
    PROCEDURE_DATE_COL, PROCEDURE_SIDE_COL, GROUPER_FORMAT_COL

JAR_FILE_PATH = f'{ROOT_DIR}/resources/jars/aimedic-grouper-assembly.jar'
SEPARATOR_CHAR = '#'
DELIMITER_CHAR = ';'

# Dataframme column names
col_aimedic_id = 'aimedicId'
col_diagnoses = 'diagnoses'
col_procedures = 'procedures'
col_grouper_result = 'grouperResult'
col_drg_cost_weight = 'drgCostWeight'
col_effective_cost_weight = 'effectiveCostWeight'
col_revision_date = 'revisionDate'
col_status = 'status'
col_used = 'used'
col_is_used = 'isUsed'
col_is_primary = 'isPrimary'
col_date_valid = 'dateValid'
col_side_valid = 'sideValid'

# Java Grouper constants
arg_filter_valid = 'filterValid'  # grouper argument, which tells the grouper it shall filter out invalid diagnoses and procedures from the case to group.
class_path_group_many = 'ch.aimedic.grouper.BatchGroupMany'


logger.debug('Testing whether Java is available ...')
subprocess.check_output(['java', '-version']).decode('utf-8')
if not os.path.exists(JAR_FILE_PATH):
    raise IOError(f"The aimedic-grouper JAR file is not available at '{JAR_FILE_PATH}")
logger.success('Java and the aimedic-grouper JAR are available')


@beartype
def group_batch_group_cases(batch_group_cases: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Groups patient cases provided in the SwissDRG Batchgrouper Format 2017 (https://grouper-docs.swissdrg.org/batchgrouper2017-format.html)
    It uses our aimedic-grouper as FAT Jar written in Scala to group the cases.

    Parameters:
    batch_group_cases (list[str]): patient cases info in SwissDrg Batchgrouper Format 2017.

    Returns:
    tuple:
        - revision_df: revision info to enter into the Database.
        - diagnoses_df: diagnoses info to enter into the Database (without revision_id).
        - procedures_df: procedures info to enter into the Database (without revision_id).
    """
    # Check for unique aimedic IDs
    #
    aimedic_ids = batch_group_cases[AIMEDIC_ID_COL].values.tolist()

    if len(set(aimedic_ids)) != (len(aimedic_ids)):
        raise ValueError('Provided cases have not unique aimedic IDs. Make sure you pass only one revision case for one patient case.')



    # Reset dataframe index and map row_index with aimedic_ids
    batch_group_cases = batch_group_cases.reset_index()
    index_aimedic_id = dict(zip(batch_group_cases['index'].values.tolist(), batch_group_cases[AIMEDIC_ID_COL].values.tolist()))

    # Fraom batch_group_cases extract the case_string

    cases_string = SEPARATOR_CHAR.join(batch_group_cases[GROUPER_FORMAT_COL])

    output = subprocess.check_output([
        'java',
        '-cp',
        JAR_FILE_PATH,
        class_path_group_many,
        cases_string,
        SEPARATOR_CHAR,
        DELIMITER_CHAR,
        arg_filter_valid]).decode('utf-8')

    # Split the captured output into lines. All but the last one contain optional log messages, whereas the last one
    # contains the JSON-output of the class we called
    lines = output.split('\n')
    grouped_cases_json = lines[-1]

    # Deserialize the output into a DataFrame
    grouped_cases_dicts = json.loads(grouped_cases_json)

    # Get the original aimedic_id
    for grouped_case in grouped_cases_dicts:
        original_aimedic_id = index_aimedic_id.get(grouped_case['aimedicId'])
        if original_aimedic_id is not None:
            grouped_case['aimedicId'] = original_aimedic_id

    complete_df = pd.DataFrame.from_dict(grouped_cases_dicts)

    # Prepare DataFrame for the revision table
    revision_df = complete_df.drop([col_diagnoses, col_procedures, col_grouper_result], axis=1)
    revision_df.rename(columns={col_aimedic_id: AIMEDIC_ID_COL,
                                col_drg_cost_weight: DRG_COST_WEIGHT_COL,
                                col_effective_cost_weight: EFFECTIVE_COST_WEIGHT_COL,
                                col_revision_date: REVISION_DATE_COL
                                }, inplace=True)

    revision_df[REVISION_DATE_COL] = pd.to_datetime(revision_df[REVISION_DATE_COL], unit='ms')

    # Prepare DataFrame for the diagnoses table
    diagnoses_df = pd.json_normalize(grouped_cases_dicts, record_path=[col_diagnoses], meta=[col_aimedic_id]) \
        .drop([col_status], axis=1)

    diagnoses_df.rename(columns={col_aimedic_id: AIMEDIC_ID_COL,
                                 col_used: IS_GROUPER_RELEVANT_COL,
                                 col_is_primary: IS_PRIMARY_COL
                                 }, inplace=True)

    diagnoses_df = diagnoses_df.reindex(columns=[AIMEDIC_ID_COL, CODE_COL, CCL_COL, IS_PRIMARY_COL, IS_GROUPER_RELEVANT_COL])

    # Prepare Dataframe for the procedure table
    # Delete empty or not defined procedures from the procedures dataframe
    grouped_cases_pd = pd.DataFrame(grouped_cases_dicts)
    grouped_cases_pd.dropna(subset=col_procedures)
    procedures_df = pd.json_normalize(grouped_cases_pd.to_dict(orient='records'), record_path=[col_procedures], meta=[col_aimedic_id]) \
        .drop([col_date_valid, col_side_valid, ], axis=1)

    procedures_df.rename(columns={col_aimedic_id: AIMEDIC_ID_COL,
                                  col_is_used: IS_GROUPER_RELEVANT_COL,
                                  col_is_primary: IS_PRIMARY_COL
                                  }, inplace=True)
    procedures_df[PROCEDURE_DATE_COL] = pd.to_datetime(procedures_df[PROCEDURE_DATE_COL]).dt.date
    procedures_df[PROCEDURE_SIDE_COL] = procedures_df[PROCEDURE_SIDE_COL].str.replace('\x00', '')  # replace empty byte string with an empty string

    # Replace NaT with NULL.
    # REFs: https://stackoverflow.com/a/42818550, https://stackoverflow.com/a/48765738
    procedures_df[PROCEDURE_DATE_COL] = procedures_df[PROCEDURE_DATE_COL].astype(object).where(procedures_df[PROCEDURE_DATE_COL].notnull(), null())

    procedures_df = procedures_df.reindex(columns=[AIMEDIC_ID_COL, CODE_COL, PROCEDURE_SIDE_COL, PROCEDURE_DATE_COL, IS_GROUPER_RELEVANT_COL, IS_PRIMARY_COL])

    return revision_df, diagnoses_df, procedures_df
