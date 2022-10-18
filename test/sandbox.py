import os
import sys

sys.path.insert(0, '/home/jovyan/work')
sys.path.insert(0, '/home/jovyan/work/src')
sys.path.insert(0, '/home/jovyan/work/src/revised_case_normalization')

from src.revised_case_normalization.py.global_configs import FileInfo, ROOT_DIR, CASE_ID_COL, COLUMNS_TO_RENAME
from src.revised_case_normalization.py.normalize import normalize

file_info = FileInfo(os.path.join(ROOT_DIR, 'raw_data/HI_Aarau_Birshof_ST. Anna.xlsx'), 'Hirslanden Aarau', '2017', ['Aarau_2017'])
cols_to_rename = dict(COLUMNS_TO_RENAME)
# Replace 'admno' with 'fall nummer'
cols_to_rename.pop('admno')
cols_to_rename['fall nummer'] = CASE_ID_COL

df_revised_case_d2d = normalize(file_info, 0)

