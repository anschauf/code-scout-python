import pandas as pd

from data_model.feature_engineering import ATC_CODES_COL


def get_atc_codes(cases: pd.DataFrame) -> pd.DataFrame:

    """This function extracts the ATC codes of each case from the 'medications column'
    of the BfS data which is formatted as such: 'ATC-CODE;annex;application;dose;unit"""

    def _get_atc_codes(row):
        medications = row['medications'].split('|')
        atc_codes = list()
        for medication in medications:
            medications_parts = medication.split(';')
            atc_codes.append(medications_parts[0])
        row[ATC_CODES_COL] = atc_codes

        return row

    cases = cases.apply(_get_atc_codes, axis=1)
    return cases
