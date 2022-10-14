import pandas as pd
from py.global_configs import *

from src.utils.dataframe_utils import validate_icd_codes, validate_chop_codes, remove_duplicated_chops, validate_pd_revised_sd


def normalize(fi: FileInfo, 
              excel_sheet_idx: int,
              *, 
              columns_mapper: dict = COLUMNS_TO_RENAME,
              columns_to_cast: dict = COLUMNS_TO_CAST,
              columns_to_lstrip: set = COLUMNS_TO_LSTRIP
              ) -> pd.DataFrame:
    """
    """
    # Read the Excel file and sheet. Cast all columns to strings, so we can format / cast the columns ourselves later on.
    # `string[pyarrow]` is an efficient way of storing strings in a DataFrame
    df = pd.read_excel(fi.path, sheet_name=fi.sheets[excel_sheet_idx], dtype='string[pyarrow]')

    # Convert all column names to lower-case, so we don't have to deal with columns named `HD Alt` vs `HD alt`
    df.columns = [c.lower() for c in df.columns]

    # Renaming columns that don't exist is a no-op. Make sure that all names actually exist
    assert(len(set(columns_mapper.keys()).difference(df.columns)) == 0)
    df.rename(columns=columns_mapper, inplace=True)
    
    assert(len(set(COLS_TO_SELECT).difference(df.columns)) == 0)
    df = df[COLS_TO_SELECT]
    n_all_rows = df.shape[0]
    print(f'Read {n_all_rows} cases for {fi.hospital_name_db} {fi.year}')
    
    # Remove rows where any value is NaN
    assert(len(set(VALIDATION_COLS).difference(df.columns)) == 0)
    df.dropna(subset=VALIDATION_COLS, inplace=True)
    n_valid_rows = df.shape[0]
    if n_valid_rows < n_all_rows:
        print(f'{n_all_rows - n_valid_rows}/{n_all_rows} rows were deleted because contained NaNs')
        
    # Cast columns to correct data type (according to DB)
    assert(len(set(columns_to_cast.keys()).difference(df.columns)) == 0)
    for col_name, col_type in columns_to_cast.items():
        df[col_name] = df[col_name].astype(col_type)  
    print(f'TYPES:\n{df.dtypes}')
    
    # Fix format of some columns
    lstrip_fun = lambda x: x.lstrip("'")
    for col_name in columns_to_lstrip:
        df[col_name] = df[col_name].apply(lstrip_fun)

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
    # compare if the primary diagnosis change or not, if changed, remove from added_icds

    # df = validate_pd_revised_sd(df, pd_col=PRIMARY_DIAGNOSIS_COL, pd_new_col=NEW_PRIMARY_DIAGNOSIS_COL, added_icd_col=ADDED_ICD_CODES, removed_icd_col=REMOVED_ICD_CODES)

    return df
    