import pandas as pd
from loguru import logger

from src.models.procedure import Procedure
from src.service.bfs_cases_db_service import get_grouped_revisions_for_sociodemographic_ids, \
    get_hospital_cases_df, get_revision_for_revision_ids
from src.service.database import Database

with Database() as db:
    hlz = get_hospital_cases_df('Hirslanden Klinik Zurich', db.session)
    sociodemographic_ids = hlz['sociodemographic_id'].values.tolist()

    logger.info('Retrieving the revision data ...')
    grouped_revisions = get_grouped_revisions_for_sociodemographic_ids(sociodemographic_ids, db.session)
    revision_ids = grouped_revisions['revision_id'].apply(lambda lst: min(lst)).values.tolist()
    revision_data = get_revision_for_revision_ids(revision_ids, db.session)
    revision_data = revision_data[['sociodemographic_id', 'revision_id', 'drg', 'mdc_partition', 'drg_cost_weight', 'effective_cost_weight', 'pccl', 'raw_pccl']]

    logger.info('Retrieving the procedure codes ...')
    all_revision_ids = set(revision_data['revision_id'].values.tolist())

    query_procedures = (
        db.session
        .query(Procedure)
        .with_entities(Procedure.code)
        .filter(Procedure.revision_id.in_(all_revision_ids))
    )

    codes = pd.read_sql(query_procedures.statement, db.session.bind)
    code_freq = codes.value_counts()

print(code_freq[:50])


