from sqlalchemy import Column, Integer, String, ForeignKey, CHAR, Date, Float, SmallInteger
from sqlalchemy import MetaData
from sqlalchemy.orm import declarative_base

metadata_obj = MetaData(schema="case_data")
Base = declarative_base(metadata=metadata_obj)


class Sociodemographics(Base):
    __tablename__ = 'sociodemographics'

    aimedic_id = Column(Integer, primary_key=True)
    hospital_id = Column(Integer, ForeignKey('hospitals.hospital_id'))
    clinic_id = Column('clinic_id', Integer, ForeignKey('chlinic.clinic_id'))
    patient_id = Column('patient_id', String(16))
    case_id = Column('case_id', String(16))
    age_years = Column('age_years', Integer)
    age_days = Column('age_days', Integer)
    gender = Column('gender', CHAR)
    duration_of_stay = Column('duration_of_stay', Integer)
    ventilation_hours = Column('ventilation_hours', Integer)
    admission_weight = Column('admission_weight', Integer)
    gestation_age = Column('gestation_age', Integer)
    admission_date = Column('admission_date', Date)
    grouper_admission_type = Column('grouper_admission_type', String(2))
    discharge_date = Column('discharge_date', Date)
    grouper_discharge_type = Column('grouper_discharge_type', String(2))
    discharge_year = Column('discharge_year', SmallInteger)
