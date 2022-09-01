import numpy as np
import venn
from src.files import load_revised_cases, load_code_scout_results, load_all_rankings


def create_rankings_of_revised_cases(
        filename_revised_cases: str,
        filename_codescout_results: str,
        dir_rankings: str
    ):

    # load revision data from DtoD
    revised_cases = load_revised_cases(filename_revised_cases)

    # Check if revision data from DtoD has KSW 2019 cases (to check if there are matching case_ids with the revision files)
    KSW = revised_cases[(revised_cases['Year'] == 2020) & (revised_cases['Clinic'] == 'KSW')]

    # renaming 'CaseId' of revision data to 'case_id' to comply with codescout rankings dataset
    revised_cases.rename(columns={'CaseId':'case_id'}, inplace =True)


    # load the KSW19 data from Codescout and extract necessary dataframe
    codescout_rankings_ksw = load_code_scout_results(filename_codescout_results)
    codescout_rankings_ksw_df = codescout_rankings_ksw[0][1]

    # sort the codescout_rankings based on probabilities

    codescout_rankings_ksw_df.sort_values(by = 'prob_most_likely_code', ascending = False)

    # make a column with a probability rank to see how high each case ranks in code scout for the comparison with DtoD
    codescout_rankings_ksw_df['probability_rank'] = codescout_rankings_ksw_df.index + 1

    # if two cases (two identical CaseID) present in codescout suggestions -> ignore the case at the moment
    # !! probably not necessary as a duplicate function is integrated in the files function
    codescout_rankings_ksw_df.drop_duplicates(subset="case_id", keep=False)

    # (and get the caseID from codescout_rankings as list)
    case_id_codescout_list = list(codescout_rankings_ksw_df['case_id'])
    revised_cases_list = list(revised_cases['case_id'])

    print("")

    # Quick check to see if any case_id from codescout appears in DtoD file
    revised_cases_ksw = revised_cases[revised_cases['case_id'].isin([case_id_codescout_list])]

    mind_bend_drg_tree = load_code_scout_results(filename_mind_bend)



    # to be modified and added later

    def list_contains(revised_cases_list, revised_cases_ksw):
        check = False
        # Iterate in the 1st list
        for item1 in revised_cases_list:
            # Iterate in the 2nd list
            for item2 in revised_cases_ksw:
                # if there is a match
                if item1 == item2:
                    check = True
                    return check
        return check


    check = any(item in revised_cases_list for item in revised_cases_kssg)

    if check:
        print("The revised_cases_list contains some elements of the revised_cases_kssg")
    else:
        print("No, revised_cases_list doesn't have any elements of the revised_cases_kssg.")

    list_contains(revised_cases_list, revised_cases_kssg)



    # go to revision cases
    # based on caseID get the index (or 'caserank_codescout') from the Codescout data

    # (iterate through revision cases (for loop) doing this:

        # for case_id in revised_cases):

        #   check if caseID (of DtoD revision data) is present in the Codescout suggestions data
        #   if yes, return the row index (i.e. position or rank bucket)

    # check the venn diagram (if possible, so we can skip making a label with rank groups) (pyvenn or venn)
    # Count the number of row index fall into different rank tiers




    # venn diagram
    # https://github.com/tctianchi/pyvenn


if __name__ == '__main__':
    create_rankings_of_revised_cases(
        dir_rankings='s3://code-scout/performance-measuring/code_rankings/mock_rankings/',
        filename_revised_cases="s3://code-scout/performance-measuring/revised_evaluation_cases.csv",
        filename_codescout_results="s3://code-scout/performance-measuring/case_rankings/DRG_tree/revisions/ksw2019/",
        filename_mindbend ="s3://code-scout/performance-measuring/code_rankings/2022-08-24_mindbend-drg-tree/drg-tree-revised-cases-total.csv"
    )


