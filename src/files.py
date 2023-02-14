import os
from os.path import basename, join, splitext

import awswrangler as wr
import numpy as np
import pandas as pd
from beartype import beartype
from loguru import logger
from tqdm import tqdm

from src.schema import case_id_col, prob_most_likely_code_col, suggested_code_probabilities_split_col, \
    suggested_code_rankings_split_col
from src.utils.general_utils import split_codes
from test.sandbox_model_case_predictions.utils import S3_PREFIX


def load_revised_cases(filename_revised_cases: str, *, verbose: bool = True) -> pd.DataFrame:
    if verbose:
        logger.info(f'Reading revised cases from {filename_revised_cases} ...')
    if filename_revised_cases.startswith(S3_PREFIX):
        revised_cases = wr.s3.read_csv(filename_revised_cases, dtype='string')
    else:
        revised_cases = pd.read_csv(filename_revised_cases, dtype='string')

    if verbose:
        logger.info(f'Read {revised_cases.shape[0]} rows')

    revised_cases[['CaseId', 'AdmNo', 'FID', 'PatID']] = revised_cases[['CaseId', 'AdmNo', 'FID', 'PatID']].fillna("")
    revised_cases['combined_id'] = revised_cases['FID'] + revised_cases['AdmNo'] + revised_cases['CaseId'] + revised_cases['PatID']

    all_case_ids = np.asarray(revised_cases['combined_id'].values)
    unique_case_ids, case_id_counts = np.unique(all_case_ids, return_counts=True)
    unique_case_ids = unique_case_ids[case_id_counts == 1]

    if all_case_ids.shape[0] != unique_case_ids.shape[0]:
        if verbose:
            logger.warning('There are duplicated case IDs in the revised cases file')

    ind_unique_case_ids = np.concatenate([np.where(revised_cases['combined_id'] == x)[0] for x in unique_case_ids])
    revised_cases = revised_cases.iloc[ind_unique_case_ids]

    revised_cases['ICD_added_split'] = revised_cases['ICD_added'].apply(split_codes)
    revised_cases['CHOP_added_split'] = revised_cases['CHOP_added'].apply(split_codes)

    return revised_cases


@beartype
def load_all_rankings(dir_rankings: str, *, verbose: bool = True) -> list[tuple[str, str, pd.DataFrame]]:
    # load rankings and store them in a tuple
    if verbose:
        logger.info(f'Listing files in {dir_rankings} ...')

    if dir_rankings.startswith(S3_PREFIX):
        all_ranking_filenames = wr.s3.list_objects(dir_rankings)
    else:
        all_ranking_filenames = [f for f in os.listdir(dir_rankings) if f.endswith('.csv')]
    if len(all_ranking_filenames) == 0:
        raise Exception(f'Found no ranking files')
    else:
        if verbose:
            logger.info(f'Found {len(all_ranking_filenames)} files')

    if verbose:
        all_ranking_filenames = tqdm(all_ranking_filenames)

    all_rankings = list()
    for filename in all_ranking_filenames:
        if filename.startswith(S3_PREFIX):
            rankings = wr.s3.read_csv(filename, sep=",", dtype='string')
            if rankings.columns.shape[0] < 2: # csv sometime seprate with ;, sometimes with ,
                rankings = wr.s3.read_csv(filename, sep=";", dtype='string')
        else:
            rankings = pd.read_csv(join(dir_rankings, filename), sep=",", dtype='string')
            if rankings.columns.shape[0] < 2: # csv sometime seprate with ;, sometimes with ,
                rankings = wr.s3.read_csv(filename, sep=";", dtype='string')
        # if UpcodingConfidenceScore column not available, create temporary one for code ranking based on the index
        if prob_most_likely_code_col not in rankings.columns:
            rankings[prob_most_likely_code_col] = 1- (rankings.index + 1)/sum(rankings.index + 1)
        if "case_id" in rankings.columns:
            rankings.rename(columns={'case_id': 'CaseId'}, inplace=True)
        if "suggested_codes" in rankings.columns:
            rankings.rename(columns={'suggested_codes': 'SuggestedCodeRankings'}, inplace=True)

        rankings = rankings.dropna(subset=['CaseId', 'UpcodingConfidenceScore'])
        rankings[prob_most_likely_code_col] = rankings[prob_most_likely_code_col].astype(float)
        rankings.drop_duplicates(subset='CaseId', inplace=True)

        all_case_ids = rankings[case_id_col].values
        unique_case_ids = np.unique(all_case_ids)
        if all_case_ids.shape[0] != unique_case_ids.shape[0]:
            raise Exception('There are duplicated case IDs in the ranked cases file')

        rankings[suggested_code_rankings_split_col] = rankings[suggested_code_rankings_split_col].apply(split_codes)
        if suggested_code_probabilities_split_col in rankings.columns:
            rankings[suggested_code_probabilities_split_col] = rankings[suggested_code_probabilities_split_col].apply(split_codes).apply(lambda probas: [float(x) for x in probas])

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

