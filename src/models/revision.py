from sqlalchemy import Column, Integer, ForeignKey, CHAR, Date, VARCHAR, FLOAT
from sqlalchemy import MetaData
from sqlalchemy.orm import declarative_base


metadata_obj = MetaData(schema="coding_revision")
Base = declarative_base(metadata=metadata_obj)


class Revision(Base):
    __tablename__ = 'revisions'

    revision_id = Column(Integer, primary_key=True)
    aimedic_id = Column(Integer, ForeignKey('case_data.sociodemographics.aimedic_id'))
    drg = Column('drg', VARCHAR)
    adrg = Column('adrg', CHAR)
    drg_cost_weight = Column('drg_cost_weight', FLOAT)
    effective_cost_weight = Column('effective_cost_weight', FLOAT)
    pccl = Column('pccl', Integer)
    revision_date = Column('revision_date', Date)
