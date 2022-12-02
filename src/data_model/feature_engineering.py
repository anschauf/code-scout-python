from sqlalchemy import Column, Integer, Float, ForeignKey, String, Boolean
from sqlalchemy import MetaData
from sqlalchemy.orm import declarative_base

from src.data_model.revision import Revision
from src.data_model.sociodemographics import Sociodemographics

# --- Define the feature columns ---
DELTA_CCL_TO_NEXT_PCCL_COL = 'delta_ccl_to_next_pccl'
HAS_VENTILATION_HOURS_COL = 'has_ventilation_hours'
VENTILATION_HOURS_ADRG_NO_A_COL = 'ventilation_hours_adrg_no_A'
VENTILATION_HOURS_BINNED_COL = 'ventilation_hours_binned'
AGE_BINNED_COL = 'age_binned'
EMERGENCY_COL = 'emergency'
HAS_HOURS_IN_ICU_COL = 'has_hours_in_icu'
HAS_NEMS_POINTS_COL = 'has_nems_points'

# ----------------------------------

metadata_obj = MetaData(schema="analytics")
Base = declarative_base(metadata=metadata_obj)


class FeatureEngineering(Base):
    __tablename__ = 'feature_engineering'

    # Define a dummy primary key, and the foreign keys to identify the case
    feateng_pk = Column(Integer, primary_key=True)
    aimedic_id = Column(String, ForeignKey(Sociodemographics.aimedic_id))
    revision_id = Column(Integer, ForeignKey(Revision.revision_id))

    # --- Features ---
    delta_ccl_to_next_pccl = Column(DELTA_CCL_TO_NEXT_PCCL_COL, Float)
    has_ventilation_hours = Column(HAS_VENTILATION_HOURS_COL, Boolean)
    ventilation_hours_binned = Column(VENTILATION_HOURS_BINNED_COL, String)
    ventilation_hours_adrg_no_A = Column(VENTILATION_HOURS_ADRG_NO_A_COL, Boolean)
    age_binned = Column(AGE_BINNED_COL, String)
    emergency = Column(EMERGENCY_COL, Boolean)
    has_hours_in_icu = Column(HAS_HOURS_IN_ICU_COL, Boolean)
    has_nems_points = Column(HAS_NEMS_POINTS_COL, Boolean)

