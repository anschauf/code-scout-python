from src.models.base import Base
from sqlalchemy import Column, Integer, String, ForeignKey, Boolean


class Clinic(Base):
    __tablename__ = 'clinics'
    clinic_id = Column(Integer, primary_key=True)
    bfs_code = Column('bfs_code', String(4))
    description = Column('description', String(500))
