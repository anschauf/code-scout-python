import os

from os.path import basename, splitext

import awswrangler as wr
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from loguru import logger


def load_revised_cases(filename_revised_cases: str) -> pd.DataFrame:
    logger.info(f'Reading revised cases from {filename_revised_cases} ...')
    revised_cases = wr.s3.read_csv(filename_revised_cases)
    logger.info(f'Read {revised_cases.shape[0]} rows')

    return revised_cases

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
        rankings = wr.s3.read_csv(filename)

        method_name = splitext(basename(filename))[0]
        hospital_year = os.path.dirname(filename).split('/')[-1]
        all_rankings.append((hospital_year, method_name, rankings))

    return all_rankings




def create_rankings_of_revised_cases(
        filename_revised_cases: str,
        filename_codescout_results: str,
    ):

    # load revision data from DtoD
    revised_cases = load_revised_cases(filename_revised_cases)
    # load the KSSG19 data from Codescout
    codescout_rankings = load_code_scout_results(filename_codescout_results)
    revised_cases['CaseId'] = revised_cases['FID']
    revised_cases['CaseId'] = revised_cases['CaseId'].fillna(0).astype(int)


    print('')
    # codescout_rankings[].sort_values(by = 'prob_most_likely_code', ascending = False)




    # sort the codescout_rankings based on probabilities and get the caseID from codescout_rankings as list
    # first the caseid from revised cases
    caseid_revised = revised_cases['CaseId'].tolist()
    caseid_codescout = codescout_rankings[0][2]['case_id'].tolist()


    # sorting once to make sure it is sorted


    # go to revision cases
    # based on caseID get the index from the Codescout data
    # iterate through revision cases (for loop) doing this:
    revised_codescout_overlap = list()

    for id in caseid_revised:
        if id in caseid_codescout and id != 0:
            revised_codescout_overlap.append(caseid_codescout.index(id))

    len(revised_codescout_overlap)
    #   check if caseID (of DtoD revision data) is present in the Codescout suggestions data
    #   if yes, return the row index (i.e. position or rank bucket)
    # if two cases (two identical CaseID) present in Codescout suggestions -> ignore the case at the moment

    # check the venn diagram (if possible, so we can skip making a label with rank groups) (pyvenn or venn)
    # Count the number of row index fall into different rank tiers


    # venn diagram
    # https://github.com/tctianchi/pyvenn

import venn
import numpy as np

labels = venn.get_labels([np.arange(10), np.arange(5, 15), np.arange(3, 8)],
                             fill=['number', 'logic'])
fig, ax = venn.venn5(labels, names=['list 1', 'list 2', 'list 3'])
fig.show()



top100 = set(np.arange(0, 100))
top1000 = set(np.arange(0, 1000))
all_cases = set(np.arange(len(caseid_codescout)))
revised_codescout_overlap = set(revised_codescout_overlap)



labels = venn.get_labels([top100, top1000, all_cases, revised_codescout_overlap],
                             fill=['number', 'logic'])
fig, ax = venn.venn5(labels, names=['top100', 'top1000', 'all_cases', 'overlap_cases'])
fig.show()




if __name__ == '__main__':
    create_rankings_of_revised_cases(
        filename_revised_cases="s3://code-scout/performance-measuring/CodeScout_GroundTruthforPerformanceMeasuring.csv",
        filename_codescout_results="s3://code-scout/performance-measuring/case_rankings/DRG_tree/revisions/ksw2019/"
    )







