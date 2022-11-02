from sqlalchemy import Column, Integer, String, ForeignKey, CHAR, Date, Float, SmallInteger, VARCHAR, FLOAT
from sqlalchemy import MetaData
from sqlalchemy.orm import declarative_base


metadata_obj = MetaData(schema="some_schema")

Base = declarative_base(metadata=metadata_obj)

class MyClass(Base):
    # will use "some_schema" by default
    __tablename__ = "sometable"

metadata_obj = MetaData(schema="coding_revision")
Base = declarative_base()

revision = Table("revision", metadata_obj,
            extend_existing=True,
            autoload_with=engine
        )