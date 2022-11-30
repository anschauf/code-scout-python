from sqlalchemy import Column, Integer, String
from sqlalchemy import MetaData
from sqlalchemy.orm import declarative_base


metadata_obj = MetaData(schema="dimension")
Base = declarative_base(metadata=metadata_obj)


class Clinic(Base):
    __tablename__ = 'clinic'
    clinic_id = Column(Integer, primary_key=True)
    description = Column('description', String(500))
