import itertools
import os
import pickle
from os.path import join

import numpy as np
import pandas as pd
import srsly
from loguru import logger
from tqdm import tqdm

from sandbox_model_case_predictions.data_handler import load_data
from sandbox_model_case_predictions.utils import get_list_of_all_predictors, get_revised_case_ids
from src import ROOT_DIR
from src.service.bfs_cases_db_service import get_bfs_cases_by_ids, get_clinics, get_codes, \
    get_grouped_revisions_for_sociodemographic_ids, \
    get_revision_for_revision_ids
from src.service.database import Database

folder_name = f'02_rf_hyperparameter_screen/01_runKSW_2020'
RESULTS_DIR = join(ROOT_DIR, 'results', 'global_performance', 'test_KSW_2020', 'n_trees_1000-max_depth_10-min_samples_leaf_400-min_samples_split_1')

with open(join(RESULTS_DIR, 'rf_cv.pkl'), 'rb') as f:
    ensemble = pickle.load(f)

logger.info('Assembling test features ...')
features_dir = join(ROOT_DIR, 'resources', 'features')
feature_filenames, encoders = get_list_of_all_predictors(load_data(only_2_rows=True), features_dir, overwrite=False, log_ignored_features=False)
DISCARDED_FEATURES = ('hospital', 'month_admission', 'month_discharge', 'year_discharge', 'hauptkostenstelle_OHE', 'vectorized_codes', 'trimmed_codes')
feature_names = sorted(list(feature_filenames.keys()))
feature_names = [feature_name for feature_name in feature_names
                 if not any(feature_name.startswith(discarded_feature) for discarded_feature in DISCARDED_FEATURES)]

REVISED_CASE_IDS_FILENAME = join(ROOT_DIR, 'resources', 'data', 'revised_case_ids.csv')
revised_cases_in_data = get_revised_case_ids(None, REVISED_CASE_IDS_FILENAME, overwrite=False)

ksw_2020 = revised_cases_in_data[(revised_cases_in_data['hospital'] == 'KSW') & (revised_cases_in_data['dischargeYear'] == 2020)].copy()
ind_hospital_leave_out = ksw_2020['index'].values

test_features = list()
for feature_name in feature_names:
    feature_filename = feature_filenames[feature_name]
    feature_values = np.load(feature_filename, mmap_mode='r', allow_pickle=False, fix_imports=False)
    test_features.append(feature_values[ind_hospital_leave_out, :])
test_features = np.hstack(test_features)

predictions = np.zeros((test_features.shape[0], len(ensemble)))
for idx, model in enumerate(ensemble):
    predictions[:, idx] = model.predict_proba(test_features)[:, 1]
predictions = np.mean(predictions, axis=1)

ksw_2020['p'] = predictions
ksw_2020.sort_values(by='p', ascending=False, inplace=True)

logger.info('Loading MindBend suggestions')
files_path = os.path.join(ROOT_DIR, 'resources', 'mind_bend_suggestions', 'ksw_2020')

all_files = os.listdir(files_path)
all_files = [os.path.join(files_path, f) for f in all_files]
all_files = [f for f in all_files if f.endswith('.json')]
logger.info(f'Found {len(all_files)} files at {files_path=}')

candidate_cases = list()

for filename in tqdm(all_files):
    json = srsly.read_json(filename)

    for case_id, suggestions in json.items():
        suggested_codes = list(itertools.chain.from_iterable(suggestions.values()))
        suggested_codes = [(s['p'], s) for s in suggested_codes]
        suggested_codes = sorted(suggested_codes, key=lambda x: -x[0])
        suggested_codes = [s[1] for s in suggested_codes]

        suggestion_types = [suggestion['codeType'] for suggestion in suggested_codes]
        is_candidate_case = any(s == 'CHOP' for s in suggestion_types[:3])

        if is_candidate_case:
            suggested_chops = [s for s in suggested_codes if s['codeType'] == 'CHOP']
            top_suggested_chops = [f"'{s['code']}' (p={s['p']:.5f}) -> {s['targetDrg']} (CW={s['targetDrgCostWeight']:.3f})" for s in suggested_chops[:10]]
            suggested_chops_str = ' | '.join(top_suggested_chops)

            candidate_cases.append((case_id, suggested_chops_str))

candidate_cases_df = pd.DataFrame(candidate_cases, columns=['id', 'suggestions'])

ksw_2020 = pd.merge(ksw_2020, candidate_cases_df, on='id', how='inner')
ksw_2020.rename(columns={'id': 'case_id'}, inplace=True)

with Database() as db:
    case_ids = ksw_2020['case_id'].values.tolist()
    logger.info(f'Retrieving the sociodemographics for {len(case_ids)} cases ...')
    sociodemographics = get_bfs_cases_by_ids(case_ids, db.session)

    sociodemographic_ids = sociodemographics['sociodemographic_id'].values.tolist()
    logger.info('Retrieving the revision data ...')
    grouped_revisions = get_grouped_revisions_for_sociodemographic_ids(sociodemographic_ids, db.session)
    revision_ids = grouped_revisions['revision_id'].apply(lambda lst: min(lst)).values.tolist()
    revision_data = get_revision_for_revision_ids(revision_ids, db.session)
    revision_data = revision_data[['sociodemographic_id', 'revision_id', 'drg', 'mdc_partition', 'drg_cost_weight', 'effective_cost_weight', 'pccl', 'raw_pccl']]

    logger.info('Retrieving the codes ...')
    codes = get_codes(revision_data[['sociodemographic_id', 'revision_id']], db.session) \
        .rename(columns={'old_pd': 'primary_diagnosis'})

    logger.info('Retrieving the clinics ...')
    clinics = get_clinics(db.session)

logger.info('Joining sociodemographics and clinics ...')
df = pd.merge(
    sociodemographics[['sociodemographic_id', 'case_id', 'clinic_id', 'age_years', 'gender', 'duration_of_stay', 'admission_date']],
    clinics[['clinic_id', 'clinic_code']],
    on='clinic_id', how='left'
)

logger.info('Joining the codes ...')
df = pd.merge(df, codes, on='sociodemographic_id', how='left')

logger.info('Joining the DRG-related info ...')
df = pd.merge(df, revision_data, on='sociodemographic_id', how='left')

logger.info('Joining CodeScout suggestions ...')
ksw_2020_export = pd.merge(ksw_2020[['case_id', 'is_revised', 'is_reviewed', 'p', 'suggestions']], df, on='case_id', how='left')

ksw_2020_export = ksw_2020_export[['case_id', 'is_revised', 'is_reviewed', 'p', 'suggestions',
                                   'age_years', 'gender', 'duration_of_stay', 'admission_date', 'clinic_code',
                                   'primary_diagnosis', 'secondary_diagnoses', 'primary_procedure', 'secondary_procedures',
                                   'drg', 'mdc_partition', 'drg_cost_weight', 'effective_cost_weight', 'pccl', 'raw_pccl']] \
    .sort_values(by='p', ascending=False)

ksw_2020_export.to_csv(os.path.join(ROOT_DIR, 'results', 'KSW_2020_export.csv'), index=False)

logger.success('done')
