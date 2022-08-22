from os.path import basename, splitext
import numpy as np
import awswrangler as wr
import pandas as pd
from matplotlib import pyplot as plt

s3_dir_rankings = 's3://code-scout/performance-measuring/mock_rankings/'
revised_cases = wr.s3.read_csv('s3://code-scout/performance-measuring/revised_evaluation_cases.csv')

# find all the rankings provided
all_ranking_filenames = wr.s3.list_objects(s3_dir_rankings)


# Data preparation (maybe possible to write this in one line)
# Once columns are prepared, it may be easier to loop through the rest (as no more Nan Values)

revised_cases['ICD_added_split'] = revised_cases['ICD_added'].apply(lambda x: x.split('|') if (isinstance(x, str)) else [])
revised_cases['CHOP_added_split'] = revised_cases['CHOP_added'].apply(lambda x: x.split('|') if (isinstance(x, str)) else [])
revised_cases['CHOP_dropped_split'] = revised_cases['CHOP_dropped'].apply(lambda x: x.split('|') if (isinstance(x, str)) else [])




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
                idx = ICD_suggested_list.index(ICD)  # error was here
                rank = idx + 1

        if rank == 'not suggest':
            rank_1_3 = 0
            rank_4_6 = 0
            rank_7_9 = 0
            rank_10 = 0
            rank_not_suggest = 1
        elif rank in [1, 2, 3]:
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

# import pandas as pd
rank_evaluation = pd.DataFrame(code_ranks, columns=['case_id', 'added_ICD', 'ICD_suggested_list', 'method_name',  'rank_1_3', 'rank_4_6', 'rank_7_9', 'rank_10', 'rank_not_suggest'])


rank_evaluation_drg_1 = rank_evaluation.loc[rank_evaluation['method_name'] == 'DRG-tree-1-prototype']
rank_evaluation_drg_2 = rank_evaluation.loc[rank_evaluation['method_name'] == 'DRG-tree-2-prototype']

drg1_rank_sum = rank_evaluation_drg_1.loc[:, ['rank_1_3', 'rank_4_6', 'rank_7_9', 'rank_10', 'rank_not_suggest']].sum(axis=0)
drg2_rank_sum = rank_evaluation_drg_2.loc[:, ['rank_1_3', 'rank_4_6', 'rank_7_9', 'rank_10', 'rank_not_suggest']].sum(axis=0)




#################
# Addition by MK
#################

drg1_rank_sum = drg1_rank_sum.reset_index(level=0)
drg2_rank_sum = drg2_rank_sum.reset_index(level=0)

drg1_rank_sum.columns.values[1] = 'Method_1'
drg2_rank_sum.columns.values[1] = 'Method_2'

import numpy as np
import matplotlib.pyplot as plt
data = [drg1_rank_sum['Method_1'],
drg2_rank_sum['Method_2']]

ranking_labels = ['rank_1_3', 'rank_4_6', 'rank_7_9', 'rank_10', 'rank_not_suggest']
X = np.arange(5)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, data[0], color = 'r', width = 0.25)
ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
ax.set_xticks(range(5))
ax.set_xticklabels(ranking_labels)

# ax.set_xticklabels(["rank_1_3", "rank_4_6", "rank_7_9", "rank_10", "rank_not_suggest"])
plt.xlabel('Ranking classes')
plt.ylabel('Frequency')
#ax.bar(X + 0.50, data[2], color = 'r', width = 0.25)
plt.savefig('bar_ranking_classes_TEST.pdf', bbox_inches='tight')


