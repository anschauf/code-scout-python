import venn
from src.files import load_revised_cases, load_code_scout_results



def create_rankings_of_revised_cases(
        filename_revised_cases: str,
        filename_codescout_results: str
    ):

    # load revision data from DtoD
    revised_cases = load_revised_cases(filename_revised_cases)
    # load the KSSG19 data from Codescout
    codescout_rankings = load_code_scout_results(filename_codescout_results)

    print('')
    codescout_rankings[].sort_values(by = 'prob_most_likely_code', ascending = False)




    # sort the codescout_rankings based on probabilities and get the caseID from codescout_rankings as list


    # sorting once to make sure it is sorted


    # go to revision cases
    # based on caseID get the index from the Codescout data

    # iterate through revision cases (for loop) doing this:

    #   check if caseID (of DtoD revision data) is present in the Codescout suggestions data
    #   if yes, return the row index (i.e. position or rank bucket)
    # if two cases (two identical CaseID) present in Codescout suggestions -> ignore the case at the moment

    # check the venn diagram (if possible, so we can skip making a label with rank groups) (pyvenn or venn)
    # Count the number of row index fall into different rank tiers


    # venn diagram
    # https://github.com/tctianchi/pyvenn


if __name__ == '__main__':
    create_rankings_of_revised_cases(
        filename_revised_cases="s3://code-scout/performance-measuring/revised_evaluation_cases.csv",
        filename_codescout_results="s3://code-scout/performance-measuring/case_rankings/DRG_tree/revisions/kssg2019/"
    )







