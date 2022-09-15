from os.path import basename, splitext
import os

import numpy as np
import pandas as pd
from loguru import logger
import awswrangler as wr
from src.schema import case_id_col, suggested_code_rankings_split_col
from src.utils import split_codes


def load_revised_cases(filename_revised_cases: str) -> pd.DataFrame:
    logger.info(f'Reading revised cases from {filename_revised_cases} ...')
    revised_cases = wr.s3.read_csv(filename_revised_cases, dtype='string')
    logger.info(f'Read {revised_cases.shape[0]} rows')

    revised_cases[['CaseId', 'AdmNo', 'FID', 'PatID']] = revised_cases[['CaseId', 'AdmNo', 'FID', 'PatID']].fillna("")
    revised_cases['combined_id'] = revised_cases['FID'] + revised_cases['AdmNo'] + revised_cases['CaseId'] + revised_cases['PatID']

    all_case_ids = np.asarray(revised_cases['combined_id'].values)
    # unique_case_ids = np.unique(all_case_ids)
    unique_case_ids, case_id_counts = np.unique(all_case_ids, return_counts=True)
    unique_case_ids = unique_case_ids[case_id_counts == 1]

    if all_case_ids.shape[0] != unique_case_ids.shape[0]:
        logger.warning('There are duplicated case IDs in the revised cases file')

    ind_unique_case_ids = np.concatenate([np.where(revised_cases['combined_id'] == x)[0] for x in unique_case_ids])
    revised_cases = revised_cases.iloc[ind_unique_case_ids]

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
        rankings = wr.s3.read_csv(filename, sep=";", dtype='string')

        all_case_ids = rankings[case_id_col].values
        unique_case_ids = np.unique(all_case_ids)
        if all_case_ids.shape[0] != unique_case_ids.shape[0]:
            raise Exception('There are duplicated case IDs in the ranked cases file')

        rankings[suggested_code_rankings_split_col] = rankings[suggested_code_rankings_split_col].apply(split_codes)

        method_name = splitext(basename(filename))[0]
        folder_name = os.path.dirname(filename).split('/')[-1]

        all_rankings.append((folder_name, method_name, rankings))

    return all_rankings


def load_code_scout_results(dir_rankings: str) -> list[(str, pd.DataFrame)]:
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
        rankings = wr.s3.read_csv(filename, sep=";", dtype='string')

        all_case_ids = rankings[case_id_col].values
        unique_case_ids = np.unique(all_case_ids)
        if all_case_ids.shape[0] != unique_case_ids.shape[0]:
            raise Exception('There are duplicated case IDs in the ranked cases file')

        method_name = splitext(basename(filename))[0]
        folder_name = os.path.dirname(filename).split('/')[-1]

        all_rankings.append((folder_name, method_name, rankings))

    return all_rankings

