from sqlalchemy import Column, Integer, String, ForeignKey, Boolean
from sqlalchemy import MetaData
from sqlalchemy.orm import declarative_base

metadata_obj = MetaData(schema="coding_revision")
Base = declarative_base(metadata=metadata_obj)


class Diagnoses(Base):
    __tablename__ = 'diagnoses'

    diagnoses_pk = Column(Integer, primary_key=True)
    aimedic_id = Column(Integer, ForeignKey('sociodemographics.aimedic_id'))
    revision_id = Column(Integer, ForeignKey('coding_revision.revisions'))

    code = Column('code', String(10))
    ccl = Column('ccl', Integer)
    is_primary = Column('is_primary', Boolean)
    is_grouper_relevant = Column('is_grouper_relevant', Boolean)
