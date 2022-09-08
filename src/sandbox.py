import os
from os.path import basename, splitext

import awswrangler as wr
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from loguru import logger
from src import venn




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

print("")


def create_rankings_of_revised_cases(
        filename_revised_cases: str,
        filename_codescout_results: str,):

    # load revision data from DtoD
    revised_cases = load_revised_cases(filename_revised_cases)
    # load the KSW2019 data from Codescout
    codescout_rankings = load_code_scout_results(filename_codescout_results)
    revised_cases['CaseId'] = revised_cases['FID']
    revised_cases['CaseId'] = revised_cases['CaseId'].fillna(0).astype(int)

    for hospital_year, method_name, rankings in codescout_rankings:
        # sort the codescout_rankings based on probabilities and get the caseID from codescout_rankings as list
        rankings.sort_values(by='prob_most_likely_code', ascending=False)
        caseid_codescout = rankings['case_id'].tolist()

        # get the caseid from revised cases as a list
        caseid_revised = revised_cases['CaseId'].tolist()

        # go to revision cases
        # based on caseID get the index from the Codescout data
        # iterate through revision cases (for loop) doing this:
        #   check if caseID (of DtoD revision data) is present in the Codescout suggestions data
        #   if yes, return the row index (i.e. position or rank bucket)
        # if two cases (two identical CaseID) present in Codescout suggestions -> ignore the case at the moment
        revised_codescout_overlap = list()
        for case in caseid_revised:
            if case in caseid_codescout and case != 0:
                revised_codescout_overlap.append(caseid_codescout.index(case))
        # check the venn diagram (if possible, so we can skip making a label with rank groups) (pyvenn or venn)
        # Count the number of row index fall into different rank tiers
        # venn diagram
        # https://github.com/tctianchi/pyvenn

        top100 = set(np.arange(0, 100))
        top1000 = set(np.arange(0, 1000))
        all_cases = set(np.arange(len(caseid_codescout)))
        revised_codescout_overlap = set(revised_codescout_overlap)

        labels = venn.get_labels([top100, top1000, all_cases, revised_codescout_overlap],
                                 fill=['number', 'logic'])
        fig, ax = venn.venn4(labels, names=['Top 100', 'Top 1000', 'All cases', 'Revised cases'])
        fig.suptitle(f'Case Ranking Tier ({hospital_year})', fontsize=40)
        fig.savefig(f'case_ranking_{hospital_year}.png', bbox_inches='tight')
        fig.show()





filename_revised_cases="s3://code-scout/performance-measuring/CodeScout_GroundTruthforPerformanceMeasuring.csv"
filename_codescout_results="s3://code-scout/performance-measuring/case_rankings/DRG_tree/revisions/ksw2019/"


#def calculate_cumulative_distribution(
#        filename_revised_cases: str,
#        filename_codescout_results: str,):

# load revision data from DtoD
revised_cases = load_revised_cases(filename_revised_cases)
# load the KSW2019 data from Codescout
codescout_rankings = load_code_scout_results(filename_codescout_results)
revised_cases['CaseId'] = revised_cases['FID']
revised_cases['CaseId'] = revised_cases['CaseId'].fillna(0).astype(int)



# fixing dataformat errors in order to convert CW_old and CW_new to floats and calculate delta_cw
revised_cases['CW_old'] = revised_cases['CW_old'].str.replace(',', '.').astype(float)

revised_cases['CW_new'] = revised_cases['CW_new'].str.replace('-', '0').astype(float)
# replace 0 with value from same index in previous column


revised_cases['delta_cw'] = revised_cases['CW_new'] - revised_cases['CW_old']


cw_new_list = revised_cases['CW_new'].tolist()
cw_old_list = revised_cases['CW_old'].tolist()

delta_cw = list()

for item1, item2 in zip(cw_new_list, cw_old_list):
    item = item1 - item2
    delta_cw.append(item)











filename_revised_cases="s3://code-scout/performance-measuring/CodeScout_GroundTruthforPerformanceMeasuring.csv"
filename_codescout_results="s3://code-scout/performance-measuring/case_rankings/DRG_tree/revisions/ksw2019/"

if __name__ == '__main__':
    create_rankings_of_revised_cases(
        filename_revised_cases="s3://code-scout/performance-measuring/CodeScout_GroundTruthforPerformanceMeasuring.csv",
        filename_codescout_results="s3://code-scout/performance-measuring/case_rankings/DRG_tree/revisions/ksw2019/")
