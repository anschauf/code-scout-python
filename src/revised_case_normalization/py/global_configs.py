import os
from dataclasses import dataclass, field

ROOT_DIR = '/home/jovyan/work/src/revised_case_normalization'


@dataclass
class FileInfo:
    path: str
    hospital_name_db: str
    year: str
    sheets: list = field(default_factory=list)
    

# List the columns to match in the DB
CASE_ID_COL = 'case_id'
PATIENT_ID_COL = 'patient_id'
GENDER_COL = 'gender'
AGE_COL = 'age_years'
CLINIC_COL = 'bfs_code'  # After joining `bfs_cases` to `clinics`, the clinic code will be stored in `bfs_code`
DURATION_OF_STAY_COL = 'duration_of_stay'
PCCL_COL = 'pccl'
PRIMARY_DIAGNOSIS_COL = 'old_pd'  # After joining `bfs_cases` to `icd_codes`, the primary diagnosis is renamed from `code` to `PD`
DRG_COL = 'drg'

# List of Sociodemographic columns necessary to retrieve SwissDRG Batchgrouper Format 2017

AGE_DAYS_COL = 'age_days'
ADMISSION_WEIGHT_COL = 'admission_weight'
GESTATION_AGE_COL = 'gestation_age'
ADMISSION_DATE_COL = 'admission_date'
ADMISSION_TYPE_COL = 'grouper_admission_type'
DISCHARGE_DATE_COL = 'discharge_date'
DISCHARGE_TYPE_COL = 'grouper_discharge_type'
VENTILATION_HOURS_COL = 'ventilation_hours'

# List of variables of the revised cases from DtoD for the SwissDRG Batchgrouper

GROUPER_PROCEDURES_COL = 'grouper_procedures'
GROUPER_DIAGNOSES_COL = 'grouper_diagnoses'
BABY_DATA_COL = 'baby_data'
PRIMARY_DIAGNOSIS_GROUPER_COL = 'primary_diagnoses'
SECONDARY_DIAGNOSES_COL = 'secondary_diagnoses'
PRIMARY_PROCEDURE_COL = 'primary_procedure'
SECONDARY_PROCEDURES_COL = 'secondary_procedures'


# List the columns which need to be imported in the DB
ADDED_ICD_CODES = 'added_icds'
REMOVED_ICD_CODES = 'removed_icds'
ADDED_CHOP_CODES = 'added_chops'
REMOVED_CHOP_CODES = 'removed_chops'
NEW_PRIMARY_DIAGNOSIS_COL = 'primary_diagnosis'
AIMEDIC_ID_COL = 'aimedic_id'


NORM_CASE_ID_COL = 'case_id_norm'

# Select the columns used to validate / match the case in the DB
VALIDATION_COLS = [
    CASE_ID_COL, PATIENT_ID_COL,
    GENDER_COL, AGE_COL,
    DURATION_OF_STAY_COL,
]

COL_SUBSET_FROM_REVISED_CASES = [PRIMARY_DIAGNOSIS_COL, NEW_PRIMARY_DIAGNOSIS_COL, CLINIC_COL, ADDED_ICD_CODES, REMOVED_ICD_CODES, ADDED_CHOP_CODES, REMOVED_CHOP_CODES]

COLS_TO_SELECT = VALIDATION_COLS + [
    NORM_CASE_ID_COL,

    # Additional columns, not yet used for validation
    PRIMARY_DIAGNOSIS_COL, NEW_PRIMARY_DIAGNOSIS_COL, CLINIC_COL, PCCL_COL, DRG_COL,

    # Columns to copy to the DB, which are not used for validation
    ADDED_ICD_CODES, REMOVED_ICD_CODES, ADDED_CHOP_CODES, REMOVED_CHOP_CODES
]


# These are the columns needed for exporting the revised cases for the Grouper (SwissDRG Batchgrouper Format 2017)
GROUPER_INPUT_BFS = [AIMEDIC_ID_COL, AGE_COL, AGE_DAYS_COL, BABY_DATA_COL, GENDER_COL,
                      ADMISSION_DATE_COL, ADMISSION_TYPE_COL, DISCHARGE_DATE_COL, DISCHARGE_TYPE_COL,
                      DURATION_OF_STAY_COL, VENTILATION_HOURS_COL]

GROUPER_INPUT_REVISED_CASES = [AIMEDIC_ID_COL, GROUPER_DIAGNOSES_COL, GROUPER_PROCEDURES_COL]



# Define a common mapping from some column names to our normalized names. 
# Mind that columns are converted to lower case when read from the file.
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
