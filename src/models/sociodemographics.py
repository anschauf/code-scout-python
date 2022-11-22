from sqlalchemy import Column, Integer, String, ForeignKey, CHAR, Date, SmallInteger, BOOLEAN, VARCHAR
from sqlalchemy import MetaData
from sqlalchemy.orm import declarative_base

metadata_obj = MetaData(schema="case_data")
Base = declarative_base(metadata=metadata_obj)


class Sociodemographics(Base):
    __tablename__ = 'sociodemographics'

    aimedic_id = Column(Integer, primary_key=True)
    hospital_id = Column(Integer, ForeignKey('hospitals.hospital_id'))
    clinic_id = Column('clinic_id', Integer, ForeignKey('clinic.clinic_id'))
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
    place_of_residence = Column('place_of_residence', String(4))
    introducing_authority = Column('introducing_authority', CHAR)
    main_cost_unit = Column('main_cost_unit', CHAR)
    health_insurance_class = Column('health_insurance_class', CHAR)
    hours_in_icu = Column('hours_in_icu', Integer)
    location_before_admission = Column('location_before_admission', String(2))
    admission_type = Column('admission_type', CHAR)
    discharge_type = Column('discharge_type', CHAR)
    location_after_discharge = Column('location_after_discharge', String(2))
    treatment_after_discharge = Column('treatment_after_discharge', CHAR)
    nems_total = Column('nems_total', String(6))
    imc_effort_points = Column('imc_effort_points', String(6))
    medications = Column('medications', VARCHAR)
    vital_status = Column('vital_status', BOOLEAN)
    multiple_birth = Column('multiple_birth', CHAR)
    congenital_malformation = Column('congenital_malformation', CHAR)
    age_of_mother = Column('age_of_mother', Integer)



