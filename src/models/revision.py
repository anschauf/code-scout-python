from sqlalchemy import BOOLEAN, CHAR, Column, Date, DECIMAL, FLOAT, ForeignKey, Integer, MetaData, VARCHAR
from sqlalchemy.ext.declarative import declarative_base

from src.models.sociodemographics import SOCIODEMOGRAPHIC_ID_COL, Sociodemographics
from src.utils.global_configs import DRG_COL, PCCL_COL

metadata_obj = MetaData(schema="coding_revision")
Base = declarative_base(metadata=metadata_obj)

REVISION_ID_COL = 'revision_id'


class Revision(Base):
    __tablename__ = 'revisions'

    revision_id = Column(REVISION_ID_COL, Integer, primary_key=True, autoincrement=True)
    sociodemographic_id = Column(SOCIODEMOGRAPHIC_ID_COL, Integer, ForeignKey(Sociodemographics.sociodemographic_id), nullable=False)
    dos_id = Column('dos_id', Integer, nullable=False)
    mdc = Column('mdc', VARCHAR(3), nullable=False)
    mdc_partition = Column('mdc_partition', CHAR)
    drg = Column(DRG_COL, VARCHAR(5))
    adrg = Column('adrg', VARCHAR(3))
    drg_cost_weight = Column('drg_cost_weight', FLOAT)
    effective_cost_weight = Column('effective_cost_weight', FLOAT)
    pccl = Column(PCCL_COL, Integer, nullable=False)
    raw_pccl = Column('raw_pccl', FLOAT, nullable=False)
    supplement_charge = Column('supplement_charge', DECIMAL(12, 2))
    supplement_charge_ppu = Column('supplement_charge_ppu', DECIMAL(12, 2))
    reviewed = Column('reviewed', BOOLEAN)
    revised = Column('revised', BOOLEAN)
    revision_date = Column('revision_date', Date, nullable=False)
