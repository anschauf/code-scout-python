import os.path

import pandas as pd
from loguru import logger

from src import ROOT_DIR
from src.models.revision import Revision
from src.service.bfs_cases_db_service import get_codes, get_sociodemographics_by_sociodemographics_ids
from src.service.database import Database
from src.utils.global_configs import GROUPER_FORMAT_COL
from src.utils.group import format_for_grouper


def export_revised_case_ids(filename: str):
    logger.info('started')

    with Database() as db:
        query_reviewed_case = db.session.query(Revision).filter(Revision.reviewed.is_(True))
        revision_info = pd.read_sql(query_reviewed_case.statement, db.session.bind)
        logger.info(f'Read {revision_info.shape[0]} reviewed cases')

        codes_info = get_codes(revision_info[['revision_id', 'sociodemographic_id']], db.session)
        logger.info('Read the codes info')

        sociodemographic_info = get_sociodemographics_by_sociodemographics_ids(revision_info['sociodemographic_id'].values.tolist(), db.session)
        logger.info('Read the sociodemographic info')

    all_reviewed_cases = pd.merge(revision_info, codes_info, on=['revision_id', 'sociodemographic_id'])
    all_reviewed_cases.rename(columns={'old_pd': 'primary_diagnosis'}, inplace=True)
    all_reviewed_cases = pd.merge(all_reviewed_cases, sociodemographic_info, on='sociodemographic_id')
    logger.info('Joined all the info')

    formatted = format_for_grouper(all_reviewed_cases, with_sociodemographic_id=False)[GROUPER_FORMAT_COL].values.tolist()
    logger.info('Converted each case to its BatchGrouper representation')

    formatted_str = '\n'.join(formatted)
    with open(filename, 'w+') as f:
        f.writelines(formatted_str)

    logger.success('done')


if __name__ == '__main__':
    export_revised_case_ids(
        filename=os.path.join(ROOT_DIR, 'results', 'reviewed_cases_for_code_scout.txt')
    )
