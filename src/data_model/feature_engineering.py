from sqlalchemy import Column, Integer, Float, ForeignKey
from sqlalchemy import MetaData
from sqlalchemy.orm import declarative_base

from src.data_model.sociodemographics import Sociodemographics
from src.data_model.revision import Revision


metadata_obj = MetaData(schema="analytics")
Base = declarative_base(metadata=metadata_obj)


class FeatureEngineering(Base):
    __tablename__ = 'feature_engineering'

    # Define a dummy primary key, and the foreign keys to identify the case
    feateng_pk = Column(Integer, primary_key=True)
    aimedic_id = Column(Integer, ForeignKey(Sociodemographics.aimedic_id))
    revision_id = Column(Integer, ForeignKey(Revision.revision_id))

    # --- Features ---
    raw_pccl = Column('raw_pccl', Float)
