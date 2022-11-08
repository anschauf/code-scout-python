import json
import subprocess

import pandas as pd
from beartype import beartype
from sqlalchemy.sql import null

from src.revised_case_normalization.py.global_configs import *

JAR_FILE_PATH = "/home/jovyan/work/resources/jars/aimedic-grouper-assembly.jar"
SEPARATOR_CHAR = "#"
DELIMITER_CHAR = ";"

@beartype
def group_batch_group_cases(batch_group_cases: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    aimedic_ids = [bgc.split(DELIMITER_CHAR)[0] for bgc in batch_group_cases]
    if not len(set(aimedic_ids)) == (len(aimedic_ids)):
        raise ValueError('Provided cases have not unique aimedic IDs. Make sure you pass only one revision case for one patient case.')

    # Send the data to the grouper
    cases_string = SEPARATOR_CHAR.join(batch_group_cases)
    grouped_cases_json = subprocess.check_output([
        'java',
        '-cp',
        JAR_FILE_PATH,
        'ch.aimedic.grouper.BatchGroupMany',
        cases_string,
        SEPARATOR_CHAR,
        DELIMITER_CHAR,
        'filterValid']).decode('utf-8')

    # Deserialize the output into a DataFrame
    grouped_cases_dicts = json.loads(grouped_cases_json)
    complete_df = pd.DataFrame.from_dict(grouped_cases_dicts)

    # Prepare DataFrame for the revision table
    revision_df = complete_df.drop(['diagnoses', 'procedures', 'grouperResult'], axis=1)
    revision_df.rename(columns={'aimedicId': AIMEDIC_ID_COL,
                                'drgCostWeight': DRG_COST_WEIGHT_COL,
                                'effectiveCostWeight': EFFECTIVE_COST_WEIGHT_COL,
                                'revisionDate': REVISION_DATE_COL
                                }, inplace=True)

    revision_df[REVISION_DATE_COL] = pd.to_datetime(revision_df[REVISION_DATE_COL], unit='ms')

    # Prepare DataFrame for the diagnoses table
    diagnoses_df = pd.json_normalize(grouped_cases_dicts, record_path=['diagnoses'], meta=['aimedicId']) \
        .drop(['status'], axis=1)

    diagnoses_df.rename(columns={'aimedicId': AIMEDIC_ID_COL,
                                 'used': IS_GROUPER_RELEVANT_COL,
                                 'isPrimary': IS_PRIMARY_COL
                                 }, inplace=True)

    diagnoses_df = diagnoses_df.reindex(columns=[AIMEDIC_ID_COL, CODE_COL, CCL_COL, IS_PRIMARY_COL, IS_GROUPER_RELEVANT_COL])

    # Prepare Dataframe for the procedure table
    procedures_df = pd.json_normalize(grouped_cases_dicts, record_path=['procedures'], meta=['aimedicId']) \
        .drop(['dateValid', 'sideValid', ], axis=1)

    procedures_df.rename(columns={'aimedicId': AIMEDIC_ID_COL,
                                  'isUsed': IS_GROUPER_RELEVANT_COL,
                                  'isPrimary': IS_PRIMARY_COL
                                  }, inplace=True)
    procedures_df[PROCEDURE_DATE_COL] = pd.to_datetime(procedures_df[PROCEDURE_DATE_COL]).dt.date
    procedures_df[PROCEDURE_SIDE_COL] = procedures_df[PROCEDURE_SIDE_COL].str.replace('\x00', '')  # replace empty byte string with an empty string

    # Replace NaT with NULL.
    # REFs: https://stackoverflow.com/a/42818550, https://stackoverflow.com/a/48765738
    procedures_df[PROCEDURE_DATE_COL] = procedures_df[PROCEDURE_DATE_COL].astype(object).where(procedures_df[PROCEDURE_DATE_COL].notnull(), null())

    procedures_df = procedures_df.reindex(columns=[AIMEDIC_ID_COL, CODE_COL, PROCEDURE_SIDE_COL, PROCEDURE_DATE_COL, IS_GROUPER_RELEVANT_COL, IS_PRIMARY_COL])

    return revision_df, diagnoses_df, procedures_df
