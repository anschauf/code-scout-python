import itertools
import os.path
import os.path
import re
import subprocess

# noinspection PyPackageRequirements
import humps  # its pypi name is pyhumps
import pandas as pd
import srsly
from beartype import beartype
from loguru import logger
from sqlalchemy.sql import null

from src import ROOT_DIR
from src.models.revision import REVISION_ID_COL
from src.models.sociodemographics import SOCIODEMOGRAPHIC_ID_COL
from src.revised_case_normalization.notebook_functions.global_configs import *

# DataFrame column names
_aimedic_id_field = 'aimedicId'
_revision_field = 'revision'
_diagnoses_field = 'diagnoses'
_procedures_field = 'procedures'
_dict_subfields = (_revision_field, _diagnoses_field, _procedures_field)


@beartype
def group_batch_group_cases(batch_group_cases: list[str],
                            separator_char: str = '#',
                            delimiter_char: str = ';'
                            ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    # Make sure that the grouper is accessible, and Java running
    jar_file_path = f'{ROOT_DIR}/resources/jars/aimedic-grouper-assembly.jar'
    is_java_running = subprocess.check_call(['java', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    if is_java_running != 0:
        raise Exception('Java is not accessible')
    if not os.path.exists(jar_file_path):
        raise IOError(f"The aimedic-grouper JAR file is not available at '{jar_file_path}")

    # Check for unique sociodemographic IDs
    sociodemographic_ids = [bgc.split(delimiter_char)[0] for bgc in batch_group_cases]
    if len(set(sociodemographic_ids)) != (len(sociodemographic_ids)):
        raise ValueError('Provided cases have not unique aimedic IDs. Make sure you pass only one revision case for one patient case.')

    # Make a string out of all the cases
    cases_string = separator_char.join(batch_group_cases)

    # Group the cases
    grouped_cases_dicts = _get_grouper_output(jar_file_path=jar_file_path,
                                              cases_string=cases_string,
                                              separator_char=separator_char,
                                              delimiter_char=delimiter_char)

    # --- Make the revision DataFrame ---
    revision_df = pd.json_normalize([d[_revision_field] for d in grouped_cases_dicts.values()])
    revision_df.columns = [humps.decamelize(col) for col in revision_df.columns]
    revision_df[REVISION_DATE_COL] = pd.to_datetime(revision_df[REVISION_DATE_COL], format='%Y%m%d')

    # --- Make the diagnoses DataFrame ---
    all_diagnosis_rows = [d[_diagnoses_field] for d in grouped_cases_dicts.values()]
    # noinspection PyTypeChecker
    diagnoses_df = pd.json_normalize(itertools.chain.from_iterable(all_diagnosis_rows))
    diagnoses_df.columns = [humps.decamelize(col) for col in diagnoses_df.columns]

    # --- Make the procedures DataFrame ---
    all_procedure_rows = [d[_procedures_field] for d in grouped_cases_dicts.values()]
    # noinspection PyTypeChecker
    procedures_df = pd.json_normalize(itertools.chain.from_iterable(all_procedure_rows))
    procedures_df.columns = [humps.decamelize(col) for col in procedures_df.columns]

    procedures_df[PROCEDURE_DATE_COL] = pd.to_datetime(procedures_df[PROCEDURE_DATE_COL], format='%Y%m%d', errors='coerce')
    # Replace NaT with NULL
    # REFs: https://stackoverflow.com/a/42818550, https://stackoverflow.com/a/48765738
    procedures_df[PROCEDURE_DATE_COL] = procedures_df[PROCEDURE_DATE_COL].astype(object).where(procedures_df[PROCEDURE_DATE_COL].notnull(), null())

    # Clear out empty strings
    procedures_df[PROCEDURE_SIDE_COL] = procedures_df[PROCEDURE_SIDE_COL].str.strip()

    # Remove primary keys from each table
    revision_df.drop(columns=[REVISION_ID_COL], inplace=True)
    diagnoses_df.drop(columns=[REVISION_ID_COL], inplace=True)
    procedures_df.drop(columns=['procedures_pk', REVISION_ID_COL], inplace=True)

    return revision_df, diagnoses_df, procedures_df


@beartype
def _get_grouper_output(*,
                        jar_file_path: str,
                        cases_string: str,
                        separator_char: str,
                        delimiter_char: str
                        ) -> dict:

    # Convert to camel-case so that we can replace the field in place
    camel_case_sociodemographic_id_col = humps.camelize(SOCIODEMOGRAPHIC_ID_COL)

    raw_output: bytes = subprocess.check_output([
        'java', '-cp', jar_file_path, 'ch.aimedic.grouper.apps.BatchGroupMany',
        cases_string, separator_char, delimiter_char])

    output = _escape_ansi(raw_output.decode('UTF-8'))

    # Split the captured output into lines, and filter only output lines, discarding log messages from the grouer
    lines = output.split('\n')
    output_lines = [line for line in lines
                    if line.startswith('{"' + _aimedic_id_field)]

    grouped_cases = dict()
    ungrouped_cases = list()
    for i, output_line in enumerate(output_lines):
        # Deserialize the output into a dict
        grouped_case_json = srsly.json_loads(output_line)

        # Get the aimedicId from the top-level dict
        aimedic_id = grouped_case_json[_aimedic_id_field]
        # Build a copy of the dict, where we insert the aimedicId in all the sub-dictionaries
        ext_grouped_case_json = dict()

        if len(grouped_case_json.keys()) != 4:
            ungrouped_case = output_lines.pop(i) # delete ungrouped case from output
            # aimedicId is actually sociodemographic_id
            ungrouped_case.replace('aimedicId', 'sociodemographic_id')
            ungrouped_cases.append(ungrouped_case)
            continue
        for key, value in grouped_case_json.items():
            if key in _dict_subfields:
                # This is a sub-dictionary, e.g., `revisions`
                if isinstance(value, dict):
                    value[camel_case_sociodemographic_id_col] = aimedic_id  # The sub-dictionary is modified in place

                elif isinstance(value, list):
                    # This is a list of dictionaries, e.g., `diagnoses` or `procedures`
                    for row in value:
                        row[camel_case_sociodemographic_id_col] = aimedic_id  # The sub-dictionary is modified in place
                        row = _fix_global_functions_list(row)


            # Store the modified dictionary
            ext_grouped_case_json[key] = value

        grouped_cases[aimedic_id] = ext_grouped_case_json
    if len(ungrouped_cases) > 0:
        logger.info(
            f'There are {len(ungrouped_cases)} cases can not be grouped. The output from grouper is: {ungrouped_cases}')

    return grouped_cases


def _fix_global_functions_list(d: dict) -> dict:
    global_functions_serialized = d.pop('globalFunctionsSerialized')
    global_functions = ' | '.join(global_functions_serialized)
    d['global_functions'] = global_functions
    return d


def _escape_ansi(line):
    ansi_escape = re.compile(r'(?:\x9B|\x1B\[)[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', line)
