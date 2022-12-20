import os
import awswrangler as wr
import pandas as pd
from loguru import logger


def read_csv_excel(path_folder, file_name):
    # process excel files
    if file_name.endswith('.xlsx'):
        # loop through all the sheet if there are more than one sheet in the excel file
        file_path = os.path.join(path_folder, file_name)
        # sheets_dict = pd.read_excel(file_path, sheet_name=None, dtype='string[pyarrow]')
        sheets_dict = wr.s3.read_excel(file_path, sheet_name=None, dtype='string[pyarrow]')

        all_sheets = []
        for name, sheet in sheets_dict.items():
            sheet['sheet'] = name
            # sheet_1 = sheet.rename(columns=lambda x: x.split('\n')[-1])
            all_sheets.append(sheet)

        df_excel = pd.concat(all_sheets)
        return df_excel

    # process csv files
    elif file_name.endswith('.csv'):
        file_path = os.path.join(path_folder, file_name)
        # df = pd.read_csv(file_path, dtype='string[pyarrow]', sep=";", encoding='ISO-8859-1')
        df_csv = wr.s3.read_csv(file_path, dtype='string[pyarrow]', sep=";", encoding='ISO-8859-1')
        return df_csv

    else:
        logger.info(f'Check the file {file_name}')


