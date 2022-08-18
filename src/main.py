from os.path import basename, splitext

import awswrangler as wr
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



for method_name, rankings in all_rankings:

     for case_index in range(revised_cases.shape[0]):
        current_case = revised_cases.iloc[case_index]

        # find matching case id in rankings if present
        # if not present, skip

        # if present split icd_added field in current_case object into single diagnoses, e.g. K5722|E870 => ['K5722',  'E870']
        # use .split('|')

        # find revised diagnoses in current ranking after also here splitting the diagnoses like before with the revised case
        # if diagnosis is present, find index where its ranked and classify it in one of the ranking labels
        # 1-3, 4-6, 7-9, 10+
        # if not present add to not suggested label

        # store in an object
        # case id, revised_icd, method name, 5 ranking labels


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









print('')