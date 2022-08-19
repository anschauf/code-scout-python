from os.path import basename, splitext

import awswrangler as wr
import pandas as pd
from matplotlib import pyplot as plt

s3_dir_rankings = 's3://code-scout/performance-measuring/mock_rankings/'
revised_cases = wr.s3.read_csv('s3://code-scout/performance-measuring/revised_evaluation_cases.csv')

# find all the rankings provided
all_ranking_filenames = wr.s3.list_objects(s3_dir_rankings)

# load rankings and store them in an tuple
all_rankings = list()
for filename in all_ranking_filenames:
    temp_data = wr.s3.read_csv(filename)
    temp_method_name = splitext(basename(filename))[0]
    all_rankings.append((temp_method_name, temp_data))

code_ranks = []

for method_name, rankings in all_rankings:

    rankings['suggested_codes_pdx_split'] = rankings['suggested_codes_pdx'].apply(lambda x: x.split('|') if (isinstance(x, str)) else [])

    revised_cases['ICD_added_split'] = revised_cases['ICD_added'].apply(lambda x: x.split('|') if (isinstance(x, str)) else [])

    for case_index in range(revised_cases.shape[0]):
        current_case = revised_cases.iloc[case_index]
        case_Id = current_case['CaseId']
        ICD_added_list = current_case['ICD_added_split']
        ranking_case_id = rankings['case_id'].tolist()

        # find matching case id in rankings if present
        # if not present, skip
        if case_Id in ranking_case_id:
            ICD_suggested_list = rankings[rankings['case_id'] == case_Id]['suggested_codes_pdx_split'].values[0]
        else:
            continue



        # if present split icd_added field in current_case object into single diagnoses, e.g. K5722|E870 => ['K5722',  'E870']
        # use .split('|')

        # find revised diagnoses in current ranking after also here splitting the diagnoses like before with the revised case
        # if diagnosis is present, find index where its ranked and classify it in one of the ranking labels
        # 1-3, 4-6, 7-9, 10+
        # if not present add to not suggested label
        for ICD in ICD_added_list:
            if ICD not in ICD_suggested_list:
                rank = 'not suggest'
            else:
                idx = ICD_added_list.index(ICD)
                rank = idx + 1

        if rank in [1,2,3]:
                rank_1_3 = 1
                rank_4_6 = 0
                rank_7_9 = 0
                rank_10 = 0
                rank_not_suggest = 0
        elif rank in [4, 5, 6]:
                rank_1_3 = 0
                rank_4_6 = 1
                rank_7_9 = 0
                rank_10 = 0
                rank_not_suggest = 0
        elif rank in [7, 8, 9]:
                rank_1_3 = 0
                rank_4_6 = 0
                rank_7_9 = 1
                rank_10 = 0
                rank_not_suggest = 0
        elif rank == 'not suggest':
                rank_1_3 = 0
                rank_4_6 = 0
                rank_7_9 = 0
                rank_10 = 0
                rank_not_suggest = 1
        else:
            rank_1_3 = 0
            rank_4_6 = 0
            rank_7_9 = 0
            rank_10 = 1
            rank_not_suggest = 0

        code_rank = [case_Id, ICD, ICD_suggested_list, method_name, rank_1_3, rank_4_6, rank_7_9, rank_10, rank_not_suggest ]
        code_ranks.append(code_rank)


        # store in an object
        # case id, revised_icd, method name, 5 ranking labels
import pandas as pd
rank_evaluation = pd.DataFrame(code_ranks, columns=['case_id', 'added_ICD', 'ICD_suggested_list', 'method_name',  'rank_1_3', 'rank_4_6', 'rank_7_9', 'rank_10', 'rank_not_suggest'])


rank_evaluation_drg_1 = rank_evaluation.loc[rank_evaluation['method_name'] == 'DRG-tree-1-prototype']
rank_evaluation_drg_2 = rank_evaluation.loc[rank_evaluation['method_name'] == 'DRG-tree-2-prototype']

drg1_rank_sum = rank_evaluation_drg_1.loc[:, ['rank_1_3', 'rank_4_6', 'rank_7_9', 'rank_10', 'rank_not_suggest']].sum(axis=0)
drg2_rank_sum = rank_evaluation_drg_2.loc[:, ['rank_1_3', 'rank_4_6', 'rank_7_9', 'rank_10', 'rank_not_suggest']].sum(axis=0)



# same result in the local folder
rank_evaluation.to_csv('rank_evaluation.csv')



# plt.figure()
# remember to rank the method from best performing method to worse performing method
# we could use just a simple approach of ranking the class label 1-3 from high to low
# plt.bar(lists of x positions of ranking labels, lists of heights of the bars, label=list of method names, color=['green', 'red'])
# plt.xlabel('Ranking classes')
# plt.ylabel('Frequency')
# plt.xticks(range(ranking_classes.shape[1]), ranking_labels)
# plt.tight_layout()
# plt.savefig(join(dir_output, f'bar_ranking_classes{tag}.pdf'), bbox_inches='tight')
# plt.close()







# print('')