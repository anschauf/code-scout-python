from sqlalchemy import Column, Integer, VARCHAR, ForeignKey, Boolean
from sqlalchemy import MetaData
from sqlalchemy.orm import declarative_base

from src.data_model.revision import REVISION_ID_COL, Revision
from src.data_model.sociodemographics import SOCIODEMOGRAPHIC_ID_COL, Sociodemographics

metadata_obj = MetaData(schema="coding_revision")
Base = declarative_base(metadata=metadata_obj)


class Diagnosis(Base):
    __tablename__ = 'diagnoses'

    diagnoses_pk = Column('diagnoses_pk', Integer, primary_key=True, autoincrement=True)
    sociodemographic_id = Column(SOCIODEMOGRAPHIC_ID_COL, Integer, ForeignKey(Sociodemographics.sociodemographic_id), nullable=False)
    revision_id = Column(REVISION_ID_COL, Integer, ForeignKey(Revision.revision_id), nullable=False)
    code = Column('code', VARCHAR(6), nullable=False)
    ccl = Column('ccl', Integer, nullable=False)
    is_primary = Column('is_primary', Boolean, nullable=False)
    is_grouper_relevant = Column('is_grouper_relevant', Boolean, nullable=False)
    global_functions = Column('global_functions', VARCHAR)
