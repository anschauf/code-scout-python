from src.models.base import Base
from sqlalchemy import Column, Integer, String, ForeignKey, CHAR, Date, Float


class BfsCase(Base):
    __tablename__ = 'bfs_cases'
    aimedic_id = Column(Integer, primary_key=True)
    hospital_id = Column(Integer, ForeignKey('hospitals.hospital_id'))
    case_id = Column('case_id', String(16))
    patient_id = Column('patient_id', String(16))
    age_years = Column('age_years', Integer)
    age_days = Column('age_days', Integer)
    gender = Column('gender', CHAR)
    duration_of_stay = Column('duration_of_stay', Integer)
    clinic_id = Column('clinic_id', Integer, ForeignKey('chlinic.clinic_id'))
    ventilation_hours = Column('ventilation_hours', Integer)
    admission_weight = Column('admission_weight', Integer)
    gestation_age = Column('gestation_age', Integer)
    admission_date = Column('admission_date', Date)
    admission_type = Column('admission_type', String(2))
    discharge_date = Column('discharge_date', Date)
    discharge_type = Column('discharge_type', String(2))
    drg = Column('drg', String(5))
    adrg = Column('adrg', String(3))
    drg_cost_weight = Column('drg_cost_weight', Float)
    effective_cost_weight = Column('effective_cost_weight', Float)
    pccl = Column('pccl', Integer)
