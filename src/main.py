from os.path import basename, splitext

import awswrangler as wr
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.utils import get_categorical_ranks

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
label_not_suggested = 'not suggest'
ranking_labels = ['rank_1_3', 'rank_4_6', 'rank_7_9', 'rank_10', 'rank_not_suggest']
for method_name, rankings in all_rankings:
    rankings['suggested_codes_pdx_split'] = rankings['suggested_codes_pdx'].apply(lambda x: x.split('|') if (isinstance(x, str)) else [])
    revised_cases['ICD_added_split'] = revised_cases['ICD_added'].apply(lambda x: x.split('|') if (isinstance(x, str)) else [])

    current_method_code_ranks = list()
    for case_index in range(revised_cases.shape[0]):
        current_case = revised_cases.iloc[case_index]
        case_Id = current_case['CaseId']
        ICD_added_list = current_case['ICD_added_split']
        ranking_case_id = rankings['case_id'].tolist()

        # find matching case id in rankings if present
        # if not present, skip
        if case_Id in ranking_case_id:
            # TODO check whether this case id is unique, if not discard case because we can not indentify it uniquely, because now we just take the first one, even if we have more
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
                rank = label_not_suggested
            else:
                idx = ICD_suggested_list.index(ICD)  # error was here
                rank = idx + 1

        current_case_label = get_categorical_ranks(rank, label_not_suggested)
        code_rank = [case_Id, ICD, ICD_suggested_list] + list(current_case_label)
        current_method_code_ranks.append(code_rank)

    code_ranks.append(pd.DataFrame(np.vstack(current_method_code_ranks), columns=['case_id', 'added_ICD', 'ICD_suggested_list'] + ranking_labels))

ranking_labels = ['rank_1_3', 'rank_4_6', 'rank_7_9', 'rank_10', 'rank_not_suggest']

# get the index ordering from highest to lowest 1-3 label rank
ind_best_to_worst_ordering = np.argsort([x[ranking_labels[0]].sum() for x in code_ranks])[::-1]

X = np.arange(len(ranking_labels))
plt.figure()
width = 0.25
offset_x = np.linspace(0, stop=width*(len(code_ranks)-1), num=len(code_ranks))
color_map = matplotlib.cm.get_cmap('rainbow')
color_map_distances = np.linspace(0, 1, len(code_ranks))
# reverse color to go from green to red
colors = [color_map(x) for x in color_map_distances]
for i_loop, i_best_model in enumerate(ind_best_to_worst_ordering):
    bar_heights = code_ranks[i_best_model][ranking_labels].sum(axis=0)
    plt.bar(X + offset_x[i_loop], bar_heights, color=colors[i_loop], width=width, label=all_rankings[i_best_model][0])
plt.xticks(range(5), ranking_labels, rotation=90)
plt.xlabel('Ranking classes')
plt.ylabel('Frequency')
plt.legend(loc='best', fancybox=True, framealpha=0.8)
plt.savefig('bar_ranking_classes_TEST_2.pdf', bbox_inches='tight')
plt.close()


