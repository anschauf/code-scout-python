import os
from dataclasses import dataclass, field


ROOT_DIR = os.getcwd().rstrip('/py')


@dataclass
class FileInfo:
    path: str
    hospital_name_db: str
    year: str
    sheets: list = field(default_factory=list)


FILES_TO_ANALYZE = {
    'Hirslanden Salem 2017': FileInfo(
         os.path.join(ROOT_DIR, 'raw_data/HI-Bern_Salem_Beau Site_Linde.xlsx'),
         'Hirslanden Salem',
         '2017',
         ['Änderungen_SA_2017']),
    
    'Hirslanden Beau Site 2017': FileInfo(
         os.path.join(ROOT_DIR, 'raw_data/HI-Bern_Salem_Beau Site_Linde.xlsx'),
         'Hirslanden Beau Site',
          '2017',
         ['Änderungen Beau Site 2017']),
        
    
    'Hirslanden Linde 2017': FileInfo(
         os.path.join(ROOT_DIR, 'raw_data/HI-Bern_Salem_Beau Site_Linde.xlsx'),
         'Hirslanden Linde',
          '2017',
         ['Änderungen_LI_2017']),
    
    'Hirslanden Linde 2018': FileInfo(
         os.path.join(ROOT_DIR, 'raw_data/HI-Bern_Salem_Beau Site_Linde.xlsx'),
         'Hirslanden Linde',
         '2018',
         ['Änderungen_LI_2018']),
    
    'Hirslanden Salem 2018': FileInfo(
         os.path.join(ROOT_DIR, 'raw_data/HI-Bern_Salem_Beau Site_Linde.xlsx'),
         'Hirslanden Salem',
         '2018',
         ['Änderungen _SA_2018']),
        
    'Hirslanden Beau Site 2018': FileInfo(
         os.path.join(ROOT_DIR, 'raw_data/HI-Bern_Salem_Beau Site_Linde.xlsx'),
         'Hirslanden Beau Site',
         '2018',
         ['Änderungen Beau Site_ 2018']),
    
    

}


# List the columns to match in the DB
CASE_ID_COL = 'case_id'
PATIENT_ID_COL = 'patient_id'
GENDER_COL = 'gender'
AGE_COL = 'age_years'
CLINIC_COL = 'bfs_code'  # After joining `bfs_cases` to `clinics`, the clinic code will be stored in `bfs_code`
DURATION_OF_STAY_COL = 'duration_of_stay'
PCCL_COL = 'pccl'
PRIMARY_DIAGNOSIS_COL = 'pd'  # After joining `bfs_cases` to `icd_codes`, the primary diagnosis is renamed from `code` to `PD`
DRG_COL = 'drg'

# List the columns which need to be imported in the DB
ADDED_ICD_CODES = 'added_icds'
REMOVED_ICD_CODES = 'removed_icds'
ADDED_CHOP_CODES = 'added_chops'
REMOVED_CHOP_CODES = 'removed_chops'
NEW_PRIMARY_DIAGNOSIS_COL = 'new_pd'

# Select the columns used to validate / match the case in the DB
VALIDATION_COLS = [
    CASE_ID_COL, PATIENT_ID_COL,
    GENDER_COL, AGE_COL,
    DURATION_OF_STAY_COL, PCCL_COL, DRG_COL,
]

COLS_TO_SELECT = VALIDATION_COLS + [
    # Additional columns, not yet used for validation
    PRIMARY_DIAGNOSIS_COL, CLINIC_COL,

    # Columns to copy to the DB, which are not used for validation
    ADDED_ICD_CODES, REMOVED_ICD_CODES, ADDED_CHOP_CODES, REMOVED_CHOP_CODES
]

# Define a common mapping from some column names to our normalized names. 
# Mind that columns are converted to lower case when read from the file.
# TODO: Zurich: Fall Nummer not admno


COLUMNS_TO_RENAME = {
    'admno': CASE_ID_COL,
    'patid': PATIENT_ID_COL,
    'geschlecht': GENDER_COL,
    'alter (jahre)': AGE_COL,
    'fab': CLINIC_COL,
    'pflegetage alt': DURATION_OF_STAY_COL,
    'pccl alt': PCCL_COL,
    'hd alt': PRIMARY_DIAGNOSIS_COL,
    'hd neu': NEW_PRIMARY_DIAGNOSIS_COL,
    'drg alt': DRG_COL,
    'hinzugefügte icd': ADDED_ICD_CODES,
    'gelöschte icd': REMOVED_ICD_CODES,
    'hinzugefügte ops': ADDED_CHOP_CODES,
    'gelöschte ops': REMOVED_CHOP_CODES,
}

COLUMNS_TO_CAST = {
    AGE_COL: int,
    DURATION_OF_STAY_COL: int,
    PCCL_COL: int,
}

COLUMNS_TO_LSTRIP = {
    PATIENT_ID_COL, 
    CASE_ID_COL,
}
