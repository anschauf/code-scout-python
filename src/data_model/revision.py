from sqlalchemy import Column, Integer, ForeignKey, CHAR, Date, VARCHAR, FLOAT, Float
from sqlalchemy import MetaData
from sqlalchemy.orm import declarative_base


metadata_obj = MetaData(schema="coding_revision")
Base = declarative_base(metadata=metadata_obj)


class Revision(Base):
    __tablename__ = 'revisions'

    revision_id = Column(Integer, primary_key=True)
    aimedic_id = Column(Integer, ForeignKey('case_data.sociodemographics.aimedic_id'))
    mdc = Column(VARCHAR(3))
    adrg = Column(CHAR)
    drg = Column(VARCHAR)
    drg_cost_weight = Column(FLOAT)
    effective_cost_weight = Column(FLOAT)
    pccl = Column(Integer)
    raw_pccl = Column(Float)
    revision_date = Column(Date)
