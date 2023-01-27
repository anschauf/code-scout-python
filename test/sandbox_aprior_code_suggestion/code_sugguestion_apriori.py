import pandas as pd
from src.service.bfs_cases_db_service import get_all_diagonosis, get_sociodemographics_for_hospital_year, \
    get_revised_case_with_codes_before_revision, get_revised_case_with_codes_after_revision
from src.service.database import Database
from efficient_apriori import apriori
from test.sandbox_model_case_predictions.data_handler import DTYPES
from src import ROOT_DIR
import os

hospital = "Kantonsspital Winterthur"
file_name = 'KSW_2020.json'
year = 2020
# get the hospital and year
dir_data = os.path.join(ROOT_DIR, 'resources', 'data'),
# df = pd.read_json(os.path.join(dir_data, 'KSW_2020.json'), lines=True, dtype=DTYPES)

#revised_cases_in_data = get_revised_case_ids(all_data, REVISED_CASE_IDS_FILENAME, overwrite=False)

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



d  = all_codes.groupby(['sociodemographic_id','revision_id'], as_index=False).agg({'code': lambda x: list(x), 'is_primary': lambda x: list(x), 'ccl': lambda x: list(x)})
case_with_case_pd = d[d['code'].apply(lambda row: case_pd_after_revision in row)]

d.to_csv("all_cases_diagnosis.csv")

# codes = [tuple(row) for row in case_with_case_pd['code'].values.tolist()]
codes_all = [tuple(row) for row in d['code'].values.tolist()]

min_support = 1.0/case_with_case_pd.shape[0]

itemsets, rules = apriori(codes_all, max_length=3)
# for max_length in [2, 3, 4, 5]:
for max_length in [2]:
    itemset_info = f'intemsets_{max_length=}'
    itemsets, rules = apriori(codes_all, max_length=max_length, min_support=min_support, min_confidence=1)

# Print a preview of the frequent itemsets
print('')