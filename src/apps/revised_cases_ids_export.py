import os.path
import sys

import pandas as pd
from loguru import logger

from src import ROOT_DIR
from src.service.bfs_cases_db_service import get_all_reviewed_cases, get_all_revised_cases, get_codes, \
    get_original_revision_id_for_sociodemographic_ids, get_revision_for_revision_ids, \
    get_sociodemographics_by_sociodemographics_ids
from src.service.database import Database
from src.utils.global_configs import GROUPER_FORMAT_COL
from src.utils.group import format_for_grouper


def export_revised_case_ids(filename: str):
    logger.info('started')

    with Database() as db:
        reviewed_revision_ids = get_all_reviewed_cases(db.session)['revision_id'].values.tolist()
        logger.info(f'Found {len(reviewed_revision_ids)} cases reviewed but not revised')

        revised_socio_ids = get_all_revised_cases(db.session)['sociodemographic_id'].values.tolist()
        revised_revision_ids = get_original_revision_id_for_sociodemographic_ids(revised_socio_ids, db.session)['revision_id'].values.tolist()
        logger.info(f'Found {len(revised_revision_ids)} cases reviewed and revised')

        all_revision_ids = sorted(list(set(reviewed_revision_ids).union(revised_revision_ids)))
        revision_info = get_revision_for_revision_ids(all_revision_ids, db.session)

        codes_info = get_codes(revision_info[['revision_id', 'sociodemographic_id']], db.session)
        logger.info('Read the codes info')

        sociodemographic_info = get_sociodemographics_by_sociodemographics_ids(revision_info['sociodemographic_id'].values.tolist(), db.session)
        logger.info('Read the sociodemographic info')

    all_reviewed_cases = pd.merge(revision_info, codes_info, on=['revision_id', 'sociodemographic_id'])
    all_reviewed_cases.rename(columns={'old_pd': 'primary_diagnosis'}, inplace=True)
    all_reviewed_cases = pd.merge(all_reviewed_cases, sociodemographic_info, on='sociodemographic_id')

    # Move all the revised cases to the top. Those cases have 'reviewed' set to False, because this is their data before revision
    all_reviewed_cases = all_reviewed_cases.sort_values(by=['reviewed', 'case_id'], ascending=[True, True]).reset_index(drop=True)

    logger.info('Joined all the info')

    formatted = format_for_grouper(all_reviewed_cases, with_sociodemographic_id=False)[GROUPER_FORMAT_COL].values.tolist()
    logger.info('Converted each case to its BatchGrouper representation')

    formatted_str = '\n'.join(formatted)
    with open(filename, 'w+') as f:
        f.writelines(formatted_str)

    logger.success(f'Written {len(formatted)} cases to {filename}')


if __name__ == '__main__':
    export_revised_case_ids(
        filename=os.path.join(ROOT_DIR, 'results', 'reviewed_cases_for_code_scout.txt')
    )

    sys.exit(0)
