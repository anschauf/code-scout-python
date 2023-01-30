import pandas as pd
from src.service.bfs_cases_db_service import get_all_diagonosis, get_sociodemographics_for_hospital_year, \
    get_revised_case_with_codes_before_revision, get_revised_case_with_codes_after_revision
from src.service.database import Database
from efficient_apriori import apriori
from test.sandbox_model_case_predictions.data_handler import DTYPES
from src import ROOT_DIR
import os
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


hospital = "Kantonsspital Winterthur"
file_name = 'KSW_2020.json'
year = 2020
# get the hospital and year
dir_data = os.path.join(ROOT_DIR, 'resources', 'data'),
# df = pd.read_json(os.path.join(dir_data, 'KSW_2020.json'), lines=True, dtype=DTYPES)

#revised_cases_in_data = get_revised_case_ids(all_data, REVISED_CASE_IDS_FILENAME, overwrite=False)

# with Database() as db:
#     all_codes = get_all_diagonosis(db.session)
#     sociodemo = get_sociodemographics_for_hospital_year(hospital_name=hospital, year=2020, session=db.session)
#     # maybe it will be better only get the sociodemographic_ids for ksw 2020
#     sociodemo_ids = sociodemo['sociodemographic_id'].tolist()
#
#     revised_cases_before_revision = get_revised_case_with_codes_before_revision(db.session)
#     revised_cases_after_revision = get_revised_case_with_codes_after_revision(db.session)
#     revised_cases_sociodemographic_id = revised_cases_before_revision['sociodemographic_id'].tolist()
#     revised_cases_sociodemographic_id_ksw = [sociodemo_id for sociodemo_id in revised_cases_sociodemographic_id if sociodemo_id in sociodemo_ids]
#
#     revised_cases_before_revision_ksw = revised_cases_before_revision[revised_cases_before_revision['sociodemographic_id'].isin(revised_cases_sociodemographic_id_ksw)]
#     revised_cases_after_revision_ksw = revised_cases_after_revision[revised_cases_after_revision['sociodemographic_id'].isin(revised_cases_sociodemographic_id_ksw)]
#
# example_case_before_revision = revised_cases_before_revision_ksw.iloc[0]
#
# case_sociodemo_id = example_case_before_revision['sociodemographic_id']
# case_pd_before_revision = example_case_before_revision["pd"]
# case_sd_before_revision = example_case_before_revision['secondary_diagnoses']
#
# example_case_after_revision = revised_cases_after_revision_ksw[revised_cases_after_revision['sociodemographic_id']==case_sociodemo_id]
# case_pd_after_revision = example_case_after_revision["pd"].values[0]
# case_sd_after_revision = example_case_after_revision['secondary_diagnoses']
#


# d  = all_codes.groupby(['sociodemographic_id','revision_id'], as_index=False).agg({'code': lambda x: list(x), 'is_primary': lambda x: list(x), 'ccl': lambda x: list(x)})
# case_with_case_pd = d[d['code'].apply(lambda row: case_pd_after_revision in row)]

# d.to_csv("all_cases_diagnosis.csv")

d = pd.read_csv(os.path.join(ROOT_DIR, 'test/sandbox_aprior_code_suggestion/all_cases_diagnosis.csv'))
# min_support = 1.0/case_with_case_pd.shape[0]

min_support = 0.001
# clear the code and save as a list
def clean_alt_list(list_):
    list_ = list_.strip('[ ]')
    list_ = list_.replace("'", '')
    list_ = list_.split(', ')

    return list_
# case_need to get the suggestions
example_ksw_2020_revised_cases_before_revision = pd.read_csv(os.path.join(ROOT_DIR, 'test/sandbox_aprior_code_suggestion/example_ksw_2020_revised_cases_before_revision.csv'))
example_ksw_2020_revised_cases_before_revision['secondary_diagnoses'] = example_ksw_2020_revised_cases_before_revision['secondary_diagnoses'].apply(clean_alt_list)
example_ksw_2020_revised_cases_before_revision

    # n = 0
for n in [1 ,2 , 3, 4]:
    row_n = example_ksw_2020_revised_cases_before_revision.iloc[n]

    revision_id = row_n['revision_id']
    sociodemographic_id = row_n['sociodemographic_id']

    case_pd = row_n['pd']
    sd = row_n['secondary_diagnoses']
    case_icds = set(sd)
    case_icds.add(case_pd)
    case_info = f'{revision_id=}_{sociodemographic_id=}'
    case_icds

    # filter out the data contain any codes from the case need suggestions
    d['code_list'] = d['code'].apply(clean_alt_list)
    d['contain_case_icds'] = d['code_list'].apply(lambda x: len(case_icds.intersection(set(x))) > 0)
    icds_subset = d[d['contain_case_icds']==True]
    print(icds_subset.shape)



    # Instantiate transaction encoder and identify unique items in transactions
    encoder = TransactionEncoder()

    # One-hot encode transactions
    subset_icds_list = icds_subset['code_list'].to_list()
    fitted = encoder.fit_transform(subset_icds_list, sparse=True)

    df_icds_onehot = pd.DataFrame.sparse.from_spmatrix(fitted, columns=encoder.columns_) # seemed to work good
    df_icds_onehot.shape

    # Compute frequent itemsets using the Apriori algorithm
    import tqdm

    frequent_itemsets = apriori(df_icds_onehot,
                                min_support = 0.001,
                                max_len = 4,
                                verbose=1,
                                low_memory=True,
                                use_colnames=True)


    # Print a preview of the frequent itemsets
    print(len(frequent_itemsets))

    # define rules based on the metrics support

    frequent_itemsets_rules_1 = association_rules(frequent_itemsets,
                                metric = "support",
                                min_threshold = 0.0015)
    # sort the rules based on the confidence and lift
    frequent_itemsets_rules_1 = frequent_itemsets_rules_1.sort_values(['confidence', 'lift'], ascending =[False, False])


    # change the antecedents and consequents from frozen set to list
    frequent_itemsets_rules_1['antecedents_ls']= frequent_itemsets_rules_1['antecedents'].apply(lambda x: list(x))
    frequent_itemsets_rules_1['consequents_ls']= frequent_itemsets_rules_1['consequents'].apply(lambda x: list(x))

    # filter rule which contains the codes from example cases
    frequent_itemsets_rules_1['case_rule_1']= frequent_itemsets_rules_1['antecedents_ls'].apply(lambda x: len(set(x).intersection(case_icds))==1)
    frequent_itemsets_rules_1['case_rule_2']= frequent_itemsets_rules_1['antecedents_ls'].apply(lambda x: len(set(x).intersection(case_icds))==2)
    frequent_itemsets_rules_1['case_rule_3']= frequent_itemsets_rules_1['antecedents_ls'].apply(lambda x: len(set(x).intersection(case_icds))==3)

    frequent_itemsets_rules_1['case_rule_result']= frequent_itemsets_rules_1['consequents_ls'].apply(lambda x: len(set(x).intersection(case_icds))==0)


    # Get the suggests based on one code, two code
    rule1 = frequent_itemsets_rules_1[(frequent_itemsets_rules_1['case_rule_1'] == True)& (frequent_itemsets_rules_1['case_rule_result'] == True)]
    rule2 = frequent_itemsets_rules_1[(frequent_itemsets_rules_1['case_rule_2'] == True)& (frequent_itemsets_rules_1['case_rule_result'] == True)]
    rule3 = frequent_itemsets_rules_1[(frequent_itemsets_rules_1['case_rule_3'] == True)& (frequent_itemsets_rules_1['case_rule_result'] == True)]

    rule2.to_csv(f'Suggest_for_{case_info}_based_on_two_codes.csv')
    rule1.to_csv(f'Suggest_for_{case_info}_based_on_one_code.csv')
    rule3.to_csv(f'Suggest_for_{case_info}_based_on_three_codes.csv')

print('')

'revision_id=816275_sociodemographic_id=816247'


# get the case before revision
