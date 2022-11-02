import subprocess
from pathlib import Path
import json
import pandas as pd

jar_file_path = "/home/jovyan/work/resources/jars/aimedic-grouper-assembly.jar"
separator_char = "#"
delimiter_char = ";"


def group_batch_group_cases(batch_group_cases: list[str]):
    """
    Groups patient cases provided in the SwissDRG Batchgrouper Format 2017 (https://grouper-docs.swissdrg.org/batchgrouper2017-format.html)
    It uses our aimedic-grouper as FAT Jar written in Scala to groupe the cases.

    Parameters:
    batch_group_cases (list[str]): patient cases info in SwissDrg Batchgrouper Format 2017.

    Returns:
    tuple:
        - revision_df: revision info to enter into the Database.
        - diagnoses_df: diagnoses info to enter into the Database (without revision_id).
        - procedures_df: procedures info to enter into the Database (without revision_id).
    """

    if not Path(jar_file_path).is_file():
        raise Exception('Aimedic grouper cannot be found in location: ' + jar_file_path)

    # Check for unique aimedic id's
    aimedicIds = [bgc.split(delimiter_char)[0] for bgc in batch_group_cases]
    if not len(set(aimedicIds)) == (len(aimedicIds)):
        raise Exception(
            'Provided cases have not unique aimedic-ids. Make sure you pass only one revision case for one patient case.')

    cases_string = separator_char.join(batch_group_cases)
    grouped_cases_json = subprocess.check_output([
        "java",
        "-cp",
        jar_file_path,
        "ch.aimedic.grouper.BatchGroupMany",
        cases_string,
        separator_char,
        delimiter_char,
        "filterValid"]).decode("utf-8")

    grouped_cases_dicts = json.loads(grouped_cases_json)
    complete_df = pd.DataFrame.from_dict(grouped_cases_dicts)

    # Prepare DataFrame for the revision table
    revision_df = complete_df.drop(['diagnoses', 'procedures', 'grouperResult'], axis=1)
    revision_df.rename(columns={'aimedicId': 'aimedic_id', 'drgCostWeight': 'drg_cost_weight',
                                'effectiveCostWeight': 'effective_cost_weight', 'revisionDate': 'revision_date'},
                       inplace=True)
    revision_df['revision_date'] = pd.to_datetime(revision_df['revision_date'], unit='ms')

    # Prepare DataFrame for the diagnoses table
    diagnoses_df = pd.json_normalize(grouped_cases_dicts, record_path=['diagnoses'], meta=['aimedicId']).drop(
        ['status'], axis=1)
    diagnoses_df.rename(columns={'aimedicId': 'aimedic_id', 'used': 'is_grouper_relevant', 'isPrimary': 'is_primary'},
                        inplace=True)
    diagnoses_df = diagnoses_df.reindex(columns=['aimedic_id', 'code', 'ccl', 'is_primary', 'is_grouper_relevant'])

    # Prepare Dataframe for the procedure table
    procedures_df = pd.json_normalize(grouped_cases_dicts, record_path=['procedures'], meta=['aimedicId']).drop(
        ['dateValid', 'sideValid', ], axis=1)
    procedures_df.rename(columns={'aimedicId': 'aimedic_id', 'isUsed': 'is_grouper_relevant', 'isPrimary': 'is_primary'}, inplace=True)
    procedures_df['date'] = pd.to_datetime(procedures_df['date'])
    procedures_df = procedures_df.reindex(columns=['aimedic_id', 'code', 'code', 'side', 'date', 'is_grouper_relevant', 'is_primary'])

    return revision_df, diagnoses_df, procedures_df
