from sqlalchemy import Column, Integer, Float, ForeignKey, String, Boolean
from sqlalchemy import MetaData
from sqlalchemy.orm import declarative_base

from src.data_model.revision import Revision
from src.data_model.sociodemographics import Sociodemographics

# --- Define the feature columns ---
DELTA_CCL_TO_NEXT_PCCL_COL = 'delta_ccl_to_next_pccl'
VENTILATION_HOURS_COL = 'ventilation_hours'
HAS_VENTILATION_HOURS_COL = 'has_ventilation_hours'
VENTILATION_HOURS_ADRG_NO_A_COL = 'ventilation_hours_adrg_no_A'
VENTILATION_HOURS_BINNED_COL = 'ventilation_hours_binned'
AGE_YEARS_COL = 'age_years'
AGE_BINNED_COL = 'age_binned'
EMERGENCY_COL = 'emergency'
NEMS_TOTAL_COL = 'nems_total'
IMC_EFFORT_POINTS_COL = 'imc_effort_points'
HAS_IMC_EFFORT_POINTS_COL = 'has_imc_effort_points'
HOURS_IN_ICU_COL = 'hours_in_icu'
HAS_HOURS_IN_ICU_COL = 'has_hours_in_icu'
HAS_NEMS_POINTS_COL = 'has_nems_points'
ATC_CODES_COL = 'atc_codes'
ADMISSION_TYPE_COL = 'admission_type'
ADMISSION_DATE_WEEKDAY_COL = 'admission_date_weekday'
ADMISSION_DATE_MONTH_COL = 'admission_date_month'
DISCHARGE_TYPE_COL = 'discharge_type'
DISCHARGE_DATE_WEEKDAY_COL = 'discharge_date_weekday'
DISCHARGE_DATE_MONTH_COL = 'discharge_date_month'

# ----------------------------------

metadata_obj = MetaData(schema="analytics")
Base = declarative_base(metadata=metadata_obj)


class FeatureEngineering(Base):
    __tablename__ = 'feature_engineering'

    # Define a dummy primary key, and the foreign keys to identify the case
    feateng_pk = Column(Integer, primary_key=True)
    sociodemographic_id = Column(Integer, ForeignKey(Sociodemographics.sociodemographics_pk))
    revision_id = Column(Integer, ForeignKey(Revision.revision_id))

    # --- Features ---
    delta_ccl_to_next_pccl = Column(DELTA_CCL_TO_NEXT_PCCL_COL, Float)
    age_years = Column(AGE_YEARS_COL, Integer)
    age_binned = Column(AGE_BINNED_COL, String)
    admission_type = Column(ADMISSION_TYPE_COL, Integer)
    emergency = Column(EMERGENCY_COL, Boolean)
    hours_in_icu = Column(HOURS_IN_ICU_COL, Integer)
    has_hours_in_icu = Column(HAS_HOURS_IN_ICU_COL, Boolean)
    nems_total = Column(NEMS_TOTAL_COL, Integer)
    has_nems_points = Column(HAS_NEMS_POINTS_COL, Boolean)
    imc_effort_points = Column(IMC_EFFORT_POINTS_COL, Integer)
    has_imc_effort_points = Column(HAS_IMC_EFFORT_POINTS_COL, Boolean)
    ventilation_hours = Column(VENTILATION_HOURS_COL, Integer)
    has_ventilation_hours = Column(HAS_VENTILATION_HOURS_COL, Boolean)
    ventilation_hours_adrg_no_A = Column(VENTILATION_HOURS_ADRG_NO_A_COL, Boolean)
    medication_atc = Column(ATC_CODES_COL, String)
    emergency = Column(EMERGENCY_COL, Boolean)
    admission_type = Column(ADMISSION_TYPE_COL, Integer)
    admission_date_weekday = Column(ADMISSION_DATE_WEEKDAY_COL, String)
    admission_date_month = Column(ADMISSION_DATE_MONTH_COL, String)
    discharge_type = Column(DISCHARGE_TYPE_COL, Integer)
    discharge_date_weekday = Column(DISCHARGE_DATE_WEEKDAY_COL, String)
    discharge_date_month = Column(DISCHARGE_DATE_MONTH_COL, String)

