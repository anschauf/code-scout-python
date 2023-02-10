import os
from os.path import join, exists

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import ttest_ind, ks_2samp
from statsmodels.stats.multitest import multipletests
from tqdm import trange

from src import ROOT_DIR
from src.apps.enrichment_utils import generate_patient_sets
from src.models.sociodemographics import SOCIODEMOGRAPHIC_ID_COL
from src.service.bfs_cases_db_service import get_sociodemographics_for_hospital_year, get_sociodemographics_for_year, \
    get_original_revision_id_for_sociodemographic_ids, get_drg_for_revision
from src.service.database import Database

hospital = 'Kantonsspital Winterthur'
year = 2019
test='t-test'
logger.info(f'Running {hospital} - {year} - {test}')
dir_output = join(ROOT_DIR, 'results', 'drg_enrichment_analysis', f'{hospital}_{year}')
if not exists(dir_output):
    os.makedirs(dir_output)

with Database() as db:
    all_cases_hospital_year = get_sociodemographics_for_hospital_year(hospital, year, db.session)
    all_cases_year = get_sociodemographics_for_year(year, db.session)

# remove patients from hospital to test
all_cases_year = all_cases_year[~all_cases_year[SOCIODEMOGRAPHIC_ID_COL].isin(all_cases_hospital_year[SOCIODEMOGRAPHIC_ID_COL].values)]

# get all chops and diagnoses
with Database() as db:
    revision_id_hopsital_year = get_original_revision_id_for_sociodemographic_ids(all_cases_hospital_year[SOCIODEMOGRAPHIC_ID_COL].tolist(), db.session)
    drgs_hospital_year = get_drg_for_revision(revision_id_hopsital_year['revision_id'].tolist(), db.session)
    revision_id_year = get_original_revision_id_for_sociodemographic_ids(all_cases_year[SOCIODEMOGRAPHIC_ID_COL].tolist(), db.session)
    drgs_year = get_drg_for_revision(revision_id_year['revision_id'].tolist(), db.session)

# get all diagnoses and chops
all_drgs, all_drgs_counts = np.unique(np.concatenate([drgs_year['drg'].values, drgs_hospital_year['drg'].values]), return_counts=True)

# generate samples for both data sets
sample_size = 2000
logger.info("generating patient sets")
patient_sets_all_data = generate_patient_sets(drgs_year.shape[0], sample_size)
patient_sets_data_to_test = generate_patient_sets(drgs_hospital_year.shape[0], sample_size, replace=True)

# initialize result fields for diags and chops
drgs_pvalues = np.ones((len(all_drgs),))
drgs_statistic = np.zeros((len(all_drgs),))
drgs_mean_all = np.zeros((len(all_drgs),))
drgs_mean_data_to_test = np.zeros((len(all_drgs),))


def get_counts(patient_sets, data, drg, col):
    counts = np.zeros((len(patient_sets), ))
    for i, set in enumerate(patient_sets):
        data_set = data.iloc[np.asarray(set)]
        counts[i] = data_set[data_set[col] == drg].shape[0]
    return counts


def run_enrichment_for_code(drg, col_name_all_data, col_name_data_to_test):
    counts_all_data_code = get_counts(patient_sets_all_data, drgs_year, drg, col_name_all_data)
    counts_data_to_test_code = get_counts(patient_sets_data_to_test, drgs_hospital_year, drg, col_name_data_to_test)
    if test == 't-test':
        stat, pval = ttest_ind(counts_all_data_code, counts_data_to_test_code)
    elif test == 'ks-test':
        stat, pval = ks_2samp(counts_all_data_code, counts_data_to_test_code)
    else:
        raise ValueError(f'Test {test} not implemented.')

    return pval, stat, counts_all_data_code, counts_data_to_test_code


for i in trange(len(all_drgs)):
    pval_drg, stat_drg, counts_all_data_drg, counts_data_to_test_drg = run_enrichment_for_code(all_drgs[i], col_name_all_data='drg', col_name_data_to_test='drg')
    drgs_pvalues[i] = pval_drg
    drgs_statistic[i] = stat_drg
    drgs_mean_all[i] = np.mean(counts_all_data_drg)
    drgs_mean_data_to_test[i] = np.mean(counts_data_to_test_drg)


_, drgs_pvalues_adj, _, _ = multipletests(drgs_pvalues, method='fdr_bh')
results_drgs = pd.DataFrame({
    'drg': all_drgs,
    'pval': drgs_pvalues,
    'stat': drgs_statistic,
    'pval_adj_fdr': drgs_pvalues_adj,
    'mean_all': drgs_mean_all,
    f'mean_{hospital}_{year}': drgs_mean_data_to_test
}).sort_values(by=['stat', 'pval'], ascending=[False, True])
results_drgs.to_csv(os.path.join(dir_output, f'{test}_drg.csv'), index=False)


print('')
