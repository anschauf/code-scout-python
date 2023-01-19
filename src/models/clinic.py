from sqlalchemy import Column, Integer, VARCHAR
from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base


metadata_obj = MetaData(schema="dimension")
Base = declarative_base(metadata=metadata_obj)


class Clinic(Base):
    __tablename__ = 'clinic'

    clinic_id = Column('clinic_id', Integer, primary_key=True, autoincrement=True)
    clinic_code = Column('clinic_code', VARCHAR(4), nullable=False)
    description = Column('description', VARCHAR(500))
