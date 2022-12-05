from sqlalchemy import Column, Integer, ForeignKey, CHAR, Date, VARCHAR, FLOAT
from sqlalchemy import MetaData
from sqlalchemy.orm import declarative_base


metadata_obj = MetaData(schema="coding_revision")
Base = declarative_base(metadata=metadata_obj)


REVISION_ID_COL = 'revision_id'


class Revision(Base):
    __tablename__ = 'revisions'

    revision_id = Column(REVISION_ID_COL, Integer, primary_key=True)
    aimedic_id = Column(Integer, ForeignKey('case_data.sociodemographics.aimedic_id'))
    drg = Column(VARCHAR)
    adrg = Column(CHAR)
    drg_cost_weight = Column(FLOAT)
    effective_cost_weight = Column(FLOAT)
    pccl = Column(Integer)
    revision_date = Column(Date)
