from sqlalchemy import Column, Integer, String, ForeignKey, CHAR, Date, Float, SmallInteger, VARCHAR, FLOAT
from sqlalchemy import MetaData, Table
from sqlalchemy.orm import declarative_base



metadata_obj = MetaData(schema="coding_revision")
Base = declarative_base()

revision = Table("revision", metadata_obj,
            extend_existing=True,
            autoload_with=engine
        )