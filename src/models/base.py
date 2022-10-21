from sqlalchemy import MetaData
from sqlalchemy.orm import declarative_base

metadata_obj = MetaData(schema="cases")
# declarative base class
Base = declarative_base(metadata=metadata_obj)