from sqlalchemy import Column, Integer, String, ForeignKey, CHAR, Date, Float, SmallInteger, VARCHAR, FLOAT
from sqlalchemy import MetaData, Table
from sqlalchemy.orm import declarative_base


metadata_obj = MetaData(schema="coding_revision")
Base = declarative_base(metadata=metadata_obj)

class Revision(Base):
    __tablename__ = 'revisions'

    revision_id = Column(Integer, primary_key=True)
    aimedic_id = Column(Integer, ForeignKey('case_data.sociodemographics'))
    drg = Column(VARCHAR)
    adrg = Column(CHAR)
    drg_cost_weight = Column(FLOAT)
    effective_cost_weight = Column(FLOAT)
    pccl = Column(Integer)
    revision_date = Column(Date)

# The Revision table class can be created/loaded from database using follow codes:
# but the engine have to be created, so I move this class to bfs_cases_db_service.py for testing

# class Revision(Base):
#    __table__ = Table(
#        "revisions",
#        metadata_obj,
#        autoload_with=engine
#    )
