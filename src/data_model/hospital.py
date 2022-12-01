from sqlalchemy import Column, Integer, String
from sqlalchemy import MetaData
from sqlalchemy.orm import declarative_base

metadata_obj = MetaData(schema="dimension")
Base = declarative_base(metadata=metadata_obj)


class Hospital(Base):
    __tablename__ = 'hospital'

    hospital_id = Column(Integer, primary_key=True)
    hospital_abbreviation = Column('hospital_abbreviation', String(5))
    hospital_name = Column('hospital_name', String(100))
    bur_nr = Column('bur_nr', nullable=True)
    company_name = Column('company_name', nullable=True)
    corporation = Column('corporation', nullable=True)
    address = Column('address', nullable=True)
