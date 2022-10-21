from sqlalchemy import CHAR, Date
from sqlalchemy import Column, Integer, String, ForeignKey, Boolean
from sqlalchemy import MetaData
from sqlalchemy.orm import declarative_base

metadata_obj = MetaData(schema="coding_revision")
Base = declarative_base(metadata=metadata_obj)


class Procedures(Base):
    __tablename__ = 'procedures'

    procedures_pk = Column(Integer, primary_key=True)
    aimedic_id = Column(Integer, ForeignKey('sociodemographics.aimedic_id'))
    revision_id = Column(Integer, ForeignKey('coding_revision.revisions'))

    code = Column('code', String(10))
    side = Column('side', CHAR)
    date = Column('date', Date)
    is_grouper_relevant = Column('is_grouper_relevant', Boolean)
    is_primary = Column('is_primary', Boolean)
