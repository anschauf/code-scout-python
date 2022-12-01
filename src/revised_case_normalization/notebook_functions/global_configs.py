import os
from dataclasses import dataclass, field


@dataclass
class FileInfo:
    path: str
    hospital_name_db: str
    year: str
    sheet: str
    

# List the columns to match in the DB
CASE_ID_COL = 'case_id'
PATIENT_ID_COL = 'patient_id'
GENDER_COL = 'gender'
AGE_COL = 'age_years'
CLINIC_COL = 'bfs_code'  # After joining `bfs_cases` to `clinics`, the clinic code will be stored in `bfs_code`
DURATION_OF_STAY_COL = 'duration_of_stay'
PCCL_COL = 'pccl'
CCL_COL = 'ccl'
PRIMARY_DIAGNOSIS_COL = 'old_pd'  # After joining `bfs_cases` to `icd_codes`, the primary diagnosis is renamed from `code` to `PD`
DRG_COL = 'drg'
DRG_COST_WEIGHT_COL = 'drg_cost_weight'
EFFECTIVE_COST_WEIGHT_COL = 'effective_cost_weight'
REVISION_DATE_COL = 'revision_date'
REVISION_ID_COL = 'revision_id'
CODE_COL = 'code'
PROCEDURE_SIDE_COL = 'side'
PROCEDURE_DATE_COL = 'date'
IS_PRIMARY_COL = 'is_primary'
IS_GROUPER_RELEVANT_COL = 'is_grouper_relevant'

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

GROUPER_FORMAT_COL = 'batchgrouper_format'

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
