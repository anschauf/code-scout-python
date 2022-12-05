from sqlalchemy import Column, Integer, String, ForeignKey, CHAR, Date, SmallInteger, BOOLEAN, VARCHAR
from sqlalchemy import MetaData
from sqlalchemy.orm import declarative_base

from src.models.clinic import Clinic
from src.models.hospital import Hospital

metadata_obj = MetaData(schema="case_data")
Base = declarative_base(metadata=metadata_obj)


AIMEDIC_ID_COL = 'aimedic_id'

# This column is the primary key in this table, but named in the following way when used as foreign key
SOCIODEMOGRAPHIC_ID_COL = 'sociodemographic_id'
SOCIODEMOGRAPHIC_PK_COL = 'sociodemographics_pk'


class Sociodemographics(Base):
    __tablename__ = 'sociodemographics'

    sociodemographics_pk = Column(SOCIODEMOGRAPHIC_PK_COL, Integer, primary_key=True, autoincrement=True)
    aimedic_id = Column(AIMEDIC_ID_COL, VARCHAR, unique=True)
    hospital_id = Column('hospital_id', Integer, ForeignKey(Hospital.hospital_id), nullable=False)
    clinic_id = Column('clinic_id', Integer, ForeignKey(Clinic.clinic_id), nullable=False)
    patient_id = Column('patient_id', String(16), nullable=False)
    case_id = Column('case_id', String(16))
    age_years = Column('age_years', Integer, nullable=False)
    age_days = Column('age_days', Integer, nullable=False)
    gender = Column('gender', CHAR, nullable=False)
    duration_of_stay = Column('duration_of_stay', Integer, nullable=False)
    ventilation_hours = Column('ventilation_hours', Integer, nullable=False)
    admission_weight = Column('admission_weight', Integer, nullable=False)
    gestation_age = Column('gestation_age', Integer, nullable=False)
    admission_date = Column('admission_date', Date, nullable=False)
    grouper_admission_type = Column('grouper_admission_type', String(2), nullable=False)
    discharge_date = Column('discharge_date', Date, nullable=False)
    grouper_discharge_type = Column('grouper_discharge_type', String(2), nullable=False)

    discharge_year = Column('discharge_year', SmallInteger, nullable=False)

    place_of_residence = Column('place_of_residence', String(4))
    introducing_authority = Column('introducing_authority', CHAR)
    main_cost_unit = Column('main_cost_unit', CHAR)
    health_insurance_class = Column('health_insurance_class', CHAR)
    hours_in_icu = Column('hours_in_icu', Integer, nullable=False)
    location_before_admission = Column('location_before_admission', String(2))
    admission_type = Column('admission_type', CHAR)
    discharge_type = Column('discharge_type', CHAR)
    location_after_discharge = Column('location_after_discharge', String(2))
    treatment_after_discharge = Column('treatment_after_discharge', CHAR)
    nems_total = Column('nems_total', String(6))
    imc_effort_points = Column('imc_effort_points', String(6))
    medications = Column('medications', VARCHAR)
    vital_status = Column('vital_status', BOOLEAN)
    multiple_birth = Column('multiple_birth', CHAR, nullable=False)
    congenital_malformation = Column('congenital_malformation', CHAR)
    age_of_mother = Column('age_of_mother', Integer)
