from src.models.base import Base
from sqlalchemy import Column, Integer, String, ForeignKey, Boolean


class IcdCode(Base):
    __tablename__ = 'icd_codes'
    aimedic_id = Column(Integer, ForeignKey('bfs_cases.aimedic_id'), primary_key=True)
    code = Column('code', String(10))
    ccl = Column('ccl', Integer)
    is_primary = Column('is_primary', Boolean)
    is_grouper_relevant = Column('is_grouper_relevant', Boolean)
