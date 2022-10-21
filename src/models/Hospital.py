from src.models.base import Base
from sqlalchemy import Column, Integer, String, ForeignKey, Boolean


class Hospital(Base):
    __tablename__ = 'hospitals'
    hospital_id = Column(Integer, primary_key=True)
    name = Column('name', String(100))
    bur_nr = Column('bur_nr', nullable=True)
    company_name = Column('company_name', nullable=True)
    corporation = Column('corporation', nullable=True)
    address = Column('address', nullable=True)
