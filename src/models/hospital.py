from sqlalchemy import Column, Integer, VARCHAR
from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base

metadata_obj = MetaData(schema="dimension")
Base = declarative_base(metadata=metadata_obj)


class Hospital(Base):
    __tablename__ = 'hospital'

    hospital_id = Column('hospital_id', Integer, primary_key=True, autoincrement=True)
    hospital_abbreviation = Column('hospital_abbreviation', VARCHAR(5), nullable=False)
    hospital_name = Column('hospital_name', VARCHAR(100), nullable=False)
    bur_nr = Column('bur_nr', VARCHAR(8))
    company_name = Column('company_name', VARCHAR(100))
    corporation = Column('corporation', VARCHAR(100))
    address = Column('address', VARCHAR(100))
