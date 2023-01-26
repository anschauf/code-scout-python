
from src.service.bfs_cases_db_service import get_all_diagonosis, get_sociodemographics_for_hospital_year, \
    get_revised_case_with_codes_before_revision, get_revised_case_with_codes_after_revision
from src.service.database import Database
from efficient_apriori import apriori


hospital = "Kantonsspital Winterthur"
year = 2020

with Database() as db:
    all_codes = get_all_diagonosis(db.session)
    sociodemo = get_sociodemographics_for_hospital_year(hospital_name=hospital, year=2020, session=db.session)
    # maybe it will be better only get the sociodemographic_ids for ksw 2020
    sociodemo_ids = sociodemo['sociodemographic_id'].tolist()

    revised_cases_before_revision = get_revised_case_with_codes_before_revision(db.session)
    revised_cases_after_revision = get_revised_case_with_codes_after_revision(db.session)
    revised_cases_sociodemographic_id = revised_cases_before_revision['sociodemographic_id'].tolist()
    revised_cases_sociodemographic_id_ksw = [sociodemo_id for sociodemo_id in revised_cases_sociodemographic_id if sociodemo_id in sociodemo_ids]

    revised_cases_before_revision_ksw = revised_cases_before_revision[revised_cases_before_revision['sociodemographic_id'].isin(revised_cases_sociodemographic_id_ksw)]
    revised_cases_after_revision_ksw = revised_cases_after_revision[revised_cases_after_revision['sociodemographic_id'].isin(revised_cases_sociodemographic_id_ksw)]

example_case_before_revision = revised_cases_before_revision_ksw.iloc[0]

case_sociodemo_id = example_case_before_revision['sociodemographic_id']
case_pd_before_revision = example_case_before_revision["pd"]
case_sd_before_revision = example_case_before_revision['secondary_diagnoses']

example_case_after_revision = revised_cases_after_revision_ksw[revised_cases_after_revision['sociodemographic_id']==case_sociodemo_id]
case_pd_after_revision = example_case_after_revision["pd"].values[0]
case_sd_after_revision = example_case_after_revision['secondary_diagnoses']



d  = all_codes.groupby('revision_id', group_keys=False)['code'].apply(list).reset_index()

case_with_case_pd = d[d['code'].apply(lambda row: case_pd_after_revision in row)]

# d.to_csv("all_cases_diagnosis.csv")

codes = [tuple(row) for row in case_with_case_pd['code'].values.tolist()]

min_support = 1.0/case_with_case_pd.shape[0]

# for max_length in [2, 3, 4, 5]:
for max_length in [2]:
    itemset_info = f'intemsets_{max_length=}'
    itemsets, rules = apriori(codes, max_length=max_length, min_support=min_support, min_confidence=1)


print('')