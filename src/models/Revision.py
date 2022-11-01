from sqlalchemy import Column, Integer, String, ForeignKey, CHAR, Date, Float, SmallInteger, VARCHAR, FLOAT
from sqlalchemy import MetaData
from sqlalchemy.orm import declarative_base



# metadata_obj = MetaData(schema="coding_revision")
Base = declarative_base()

class Revision(Base):
    __tablename__ = 'revision'
    revision_id = Column(Integer, primary_key=True)
    aimedic_id = Column(Integer, ForeignKey('case_data.sociodemographics'))
    drg = Column(VARCHAR)
    adrg = Column(CHAR)
    drg_cost_weight = Column(FLOAT)
    effective_cost_weight = Column(FLOAT)
    pccl = Column(Integer)
    revision_date = Column(Date)


from marshmallow_sqlalchemy import SQLAlchemyAutoSchema, auto_field

class RevisionSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Revision
        load_instance = True
