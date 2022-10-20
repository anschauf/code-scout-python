import pandas as pd

from src.models.BfsCase import BfsCase
from src.service.bfs_cases_db_service import session, get_hospital_year_cases
from sqlalchemy.dialects.postgresql import BIGINT




query = session.query(BfsCase) \
    .filter(BfsCase.case_id.cast(BIGINT) == 41418378)

result = pd.read_sql(query.statement, session.bind)

# results = (
#     session
#     .query(BfsCase)
# )


print('')
