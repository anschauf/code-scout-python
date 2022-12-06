from sqlalchemy import MetaData, CHAR, Date, Column, Integer, VARCHAR, ForeignKey, Boolean, DECIMAL
from sqlalchemy.orm import declarative_base

from src.data_model.revision import REVISION_ID_COL, Revision
from src.data_model.sociodemographics import SOCIODEMOGRAPHIC_ID_COL, Sociodemographics

metadata_obj = MetaData(schema='coding_revision')
Base = declarative_base(metadata=metadata_obj)


class Procedure(Base):
    __tablename__ = 'procedures'

    procedures_pk = Column('procedures_pk', Integer, primary_key=True, autoincrement=True)
    sociodemographic_id = Column(SOCIODEMOGRAPHIC_ID_COL, Integer, ForeignKey(Sociodemographics.sociodemographics_pk), nullable=False)
    revision_id = Column(REVISION_ID_COL, Integer, ForeignKey(Revision.revision_id), nullable=False)

    code = Column('code', VARCHAR(8), nullable=False)
    side = Column('side', CHAR)
    date = Column('date', Date)
    is_grouper_relevant = Column('is_grouper_relevant', Boolean, nullable=False)
    is_primary = Column('is_primary', Boolean, nullable=False)
    global_functions = Column('global_functions', VARCHAR)
    supplement_charge = Column('supplement_charge', DECIMAL(12, 2))
    supplement_charge_ppu = Column('supplement_charge_ppu', DECIMAL(12, 2))
