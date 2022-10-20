from src.models.base import Base
from sqlalchemy import Column, Integer, String, ForeignKey, Boolean, CHAR, Date


class ChopCode(Base):
    __tablename__ = 'chop_codes'
    aimedic_id = Column(Integer, ForeignKey('bfs_cases.aimedic_id'), primary_key=True)
    code = Column('code', String(8))
    side = Column('side', CHAR)
    date = Column('date', Date)
    is_grouper_relevant = Column('is_grouper_relevant', Boolean)
    is_primary = Column('is_primary', Boolean)
