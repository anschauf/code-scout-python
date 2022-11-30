from sqlalchemy import Column, Integer, Float, ForeignKey
from sqlalchemy import MetaData
from sqlalchemy.orm import declarative_base

metadata_obj = MetaData(schema="analytics")
Base = declarative_base(metadata=metadata_obj)


class FeatureEngineering(Base):
    __tablename__ = 'feature_engineering'

    feateng_pk = Column(Integer, primary_key=True)
    # aimedic_id = Column(Integer, ForeignKey('case_data.sociodemographics.aimedic_id'))
    # revision_id = Column(Integer, ForeignKey('coding_revision.revisions.revision_id'))

    raw_pccl = Column('raw_pccl', Float)
