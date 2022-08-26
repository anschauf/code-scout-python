from os.path import basename, splitext

import numpy as np
import pandas as pd
from loguru import logger
import awswrangler as wr

from src.utils import split_codes


def load_revised_cases(filename_revised_cases: str) -> pd.DataFrame:
    logger.info(f'Reading revised cases from {filename_revised_cases} ...')
    revised_cases = wr.s3.read_csv(filename_revised_cases)
    logger.info(f'Read {revised_cases.shape[0]} rows')

    all_case_ids = revised_cases['CaseId'].values
    unique_case_ids = np.unique(all_case_ids)
    if all_case_ids.shape[0] != unique_case_ids.shape[0]:
        raise Exception('There are duplicated case IDs in the revised cases file')

    revised_cases['ICD_added_split'] = revised_cases['ICD_added'].apply(split_codes)
    revised_cases['CHOP_added_split'] = revised_cases['CHOP_added'].apply(split_codes)
    revised_cases['CHOP_dropped_split'] = revised_cases['CHOP_dropped'].apply(split_codes)

    return revised_cases


def load_all_rankings(dir_rankings: str) -> list[(str, pd.DataFrame)]:
    # load rankings and store them in a tuple
    logger.info(f'Listing files in {dir_rankings} ...')
    all_ranking_filenames = wr.s3.list_objects(dir_rankings)
    if len(all_ranking_filenames) == 0:
        raise Exception(f'Found no ranking files')
    else:
        logger.info(f'Found {len(all_ranking_filenames)} files')

    all_rankings = list()
    for filename in all_ranking_filenames:
        logger.info(f'Reading {filename} ...')
        rankings = wr.s3.read_csv(filename)

        all_case_ids = rankings['case_id'].values
        unique_case_ids = np.unique(all_case_ids)
        if all_case_ids.shape[0] != unique_case_ids.shape[0]:
            raise Exception('There are duplicated case IDs in the ranked cases file')

        rankings['suggested_codes_pdx_split'] = rankings['suggested_codes_pdx'].apply(split_codes)

        method_name = splitext(basename(filename))[0]

        all_rankings.append((method_name, rankings))

    return all_rankings
