import pandas as pd
from py.global_configs import *



def normalize(fi:FileInfo, excel_sheet_idx: int,
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

    return df 
    