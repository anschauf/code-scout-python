import numpy as np
import pandas as pd
from beartype import beartype
from loguru import logger

from src.revised_case_normalization.notebook_functions.revised_case_files_info import  FileInfo
from src.revised_case_normalization.notebook_functions.global_configs import  COLUMNS_TO_RENAME, \
    COLUMNS_TO_LSTRIP, COLUMNS_TO_CAST, DURATION_OF_STAY_COL, NORM_CASE_ID_COL, VALIDATION_COLS, ADDED_ICD_CODES, \
    REMOVED_ICD_CODES, ADDED_CHOP_CODES, REMOVED_CHOP_CODES, PRIMARY_DIAGNOSIS_COL, NEW_PRIMARY_DIAGNOSIS_COL, \
    COLS_TO_SELECT, CASE_ID_COL
from src.utils.dataframe_utils import validate_icd_codes, validate_chop_codes, remove_duplicated_chops, \
    validate_pd_revised_sd


@beartype
def normalize(fi: FileInfo,
              *,
              columns_mapper: dict = COLUMNS_TO_RENAME,
              columns_to_lstrip: set = COLUMNS_TO_LSTRIP,
              columns_to_cast: dict = COLUMNS_TO_CAST,
              ) -> pd.DataFrame:
    """Given an Excel sheet containing an almost standardized schema for revising cases (which is manually filled in by
    coders), this function normalizes that content so that it can be further validated, analyzed, and loaded into the
    DB.

    @param fi: An instance of FileInfo, which contains the filename and the sheet name to analyze.
    @param columns_mapper: A dictionary of columns to rename. All column names are lower-cased before being mapped. A
        default is provided.
    @param columns_to_lstrip: A set of columns on which to apply `.lstrip("'")`, which removes the leading apostrophe
        symbol.
    @param columns_to_cast: A dictionary of the columns to cast to a different type, from string. A default is provided.

    @return: A pandas DataFrame, with the normalized column names and data. If any row is discarded, it is logged.
    """
    # Read the Excel file and sheet. Cast all columns to strings, so we can format / cast the columns ourselves later on.
    # `string[pyarrow]` is an efficient way of storing strings in a DataFrame
    df = pd.read_excel(fi.path, sheet_name=fi.sheet, dtype='string[pyarrow]')
    n_all_rows = df.shape[0]
    logger.info(f'Read {n_all_rows} cases for {fi.hospital_name_db} {fi.year}')

    # Convert all column names to lower-case, so we don't have to deal with columns named `HD Alt` vs `HD alt`
    df.columns = [c.lower() for c in df.columns]

    # Renaming columns that don't exist is a no-op. Make sure that all names actually exist
    non_existing_columns_to_rename = set(columns_mapper.keys()).difference(df.columns)
    if len(non_existing_columns_to_rename) > 0:
        raise ValueError(f'The following columns to rename did not exist: {sorted(list(non_existing_columns_to_rename))}. Available columns are {list(df.columns)}')
    df.rename(columns=columns_mapper, inplace=True)

    # Fix unavailable duration of stay
    df[DURATION_OF_STAY_COL] = df[DURATION_OF_STAY_COL].replace('n.ü.', np.nan)

    # Discard rows where any value on any validation col is empty
    non_existing_validation_cols = set(VALIDATION_COLS).difference(df.columns)
    if len(non_existing_validation_cols) > 0:
        raise ValueError(f'The following columns to validate did not exist: {sorted(list(non_existing_validation_cols))}')

    df.dropna(subset=VALIDATION_COLS, inplace=True)

    n_valid_rows = df.shape[0]
    if n_valid_rows < n_all_rows:
        logger.info(f'{n_all_rows - n_valid_rows}/{n_all_rows} rows were deleted because contained NaNs')

    # Fix format of some columns
    lstrip_fun = lambda x: x.lstrip("'")
    for col_name in columns_to_lstrip:
        df[col_name] = df[col_name].apply(lstrip_fun)

    # Duplicate the case ID column which does not have leading zeros
    df[NORM_CASE_ID_COL] = df[CASE_ID_COL].apply(remove_leading_zeros)

    for col_name, col_type in columns_to_cast.items():
        df[col_name] = df[col_name].astype(col_type)
    logger.info(f'TYPES:\n{df.dtypes}')

    # Split ICD and CHOP columns into list[str]
    for code_col_to_fix in (ADDED_ICD_CODES, REMOVED_ICD_CODES, ADDED_CHOP_CODES, REMOVED_CHOP_CODES):
        df[code_col_to_fix] = df[code_col_to_fix].fillna('').str.split(',')

    # Validate ICD and CHOP codes
    df = validate_icd_codes(df, icd_codes_col=ADDED_ICD_CODES, output_icd_codes_col=ADDED_ICD_CODES)
    df = validate_icd_codes(df, icd_codes_col=REMOVED_ICD_CODES, output_icd_codes_col=REMOVED_ICD_CODES)
    df = validate_chop_codes(df, chop_codes_col=ADDED_CHOP_CODES, output_chop_codes_col=ADDED_CHOP_CODES)
    df = validate_chop_codes(df, chop_codes_col=REMOVED_CHOP_CODES, output_chop_codes_col=REMOVED_CHOP_CODES)

    # Remove CHOP codes which appear in both added and removed lists
    df = remove_duplicated_chops(df,
                                 added_chops_col=ADDED_CHOP_CODES, cleaned_added_chops_col=ADDED_CHOP_CODES,
                                 removed_chops_col=REMOVED_CHOP_CODES, cleaned_removed_chops_col=REMOVED_CHOP_CODES)

    # Compare if the primary diagnosis changed or not. If so, remove it from added ICD and removed ICD lists
    df = validate_pd_revised_sd(df,
                                pd_col=PRIMARY_DIAGNOSIS_COL,
                                pd_new_col=NEW_PRIMARY_DIAGNOSIS_COL,
                                added_icd_col=ADDED_ICD_CODES,
                                removed_icd_col=REMOVED_ICD_CODES)

    # Select columns
    non_existing_columns_to_select = set(COLS_TO_SELECT).difference(df.columns)
    if len(non_existing_columns_to_select) > 0:
        raise ValueError(f'The following columns to select did not exist: {sorted(list(non_existing_columns_to_select))}')
    df = df[COLS_TO_SELECT]

    # Remove duplicated cases
    n_rows_with_duplicates = df.shape[0]
    df.drop_duplicates(subset=[NORM_CASE_ID_COL], keep=False, inplace=True)
    df.drop_duplicates(subset=VALIDATION_COLS, keep=False, inplace=True)
    n_rows_without_duplicates = df.shape[0]
    if n_rows_without_duplicates != n_rows_with_duplicates:
        logger.info(f'Removed {n_rows_with_duplicates - n_rows_without_duplicates} rows containing duplicates')

    logger.success('Completed validation')
    return df


def remove_leading_zeros(s: str) -> str:
    return s.lstrip('0')

