from sqlalchemy import Column, Integer, VARCHAR
from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base


metadata_obj = MetaData(schema="dimension")
Base = declarative_base(metadata=metadata_obj)


class DurationOfStay(Base):
    __tablename__ = 'duration_of_stay'

    dos_id = Column('dos_id', Integer, primary_key=True, autoincrement=True)
    dos_legacy_code = Column('dos_legacy_code', VARCHAR(2), nullable=False)
    description = Column('description', VARCHAR(500))
