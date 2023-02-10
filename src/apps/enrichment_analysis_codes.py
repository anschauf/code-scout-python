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
    get_original_revision_id_for_sociodemographic_ids, get_codes
from src.service.database import Database

hospital = 'Kantonsspital Winterthur'
year = 2019
test='t-test'
dir_output = join(ROOT_DIR, 'results', 'code_enrichment_analysis', f'{hospital}_{year}')
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
    codes_hospital_year = get_codes(revision_id_hopsital_year, db.session)
    revision_id_year = get_original_revision_id_for_sociodemographic_ids(all_cases_year[SOCIODEMOGRAPHIC_ID_COL].tolist(), db.session)
    codes_year = get_codes(revision_id_year, db.session)


def combine_diagnoses(row):
    combined = row['secondary_diagnoses']
    combined.append(row['old_pd'])
    return combined

def combine_chops(row):
    combined = row['secondary_procedures']
    combined.append(row['primary_procedure'])
    combined_split = [x.split(':')[0] for x in combined]
    if '' in combined_split:
        combined_split.remove('')
    return combined_split


codes_hospital_year['all_diagnoses'] = codes_hospital_year.apply(combine_diagnoses, axis=1)
codes_year['all_diagnoses'] = codes_year.apply(combine_diagnoses, axis=1)
codes_hospital_year['all_procedures'] = codes_hospital_year.apply(combine_chops, axis=1)
codes_year['all_procedures'] = codes_year.apply(combine_chops, axis=1)

# get all diagnoses and chops
all_diagnoses, all_diagnoses_counts = np.unique(np.concatenate([np.concatenate(codes_hospital_year['all_diagnoses'].values), np.concatenate(codes_year['all_diagnoses'].values)]), return_counts=True)
all_chops, all_chops_counts = np.unique(np.concatenate([np.concatenate(codes_hospital_year['all_procedures'].values), np.concatenate(codes_year['all_procedures'].values)]), return_counts=True)

# generate samples for both data sets
sample_size = 1000
logger.info("generating patient sets")
patient_sets_all_data = generate_patient_sets(codes_year.shape[0], sample_size)
patient_sets_data_to_test = generate_patient_sets(codes_hospital_year.shape[0], sample_size)

# initialize result fields for diags and chops
diags_pvalues = np.ones((len(all_diagnoses), ))
diags_statistic = np.zeros((len(all_diagnoses), ))
diags_mean_all = np.zeros((len(all_diagnoses), ))
diags_mean_data_to_test = np.zeros((len(all_diagnoses), ))

chops_pvalues = np.ones((len(all_chops), ))
chops_statistic = np.zeros((len(all_chops), ))
chops_mean_all = np.zeros((len(all_chops), ))
chops_mean_data_to_test = np.zeros((len(all_chops), ))


def get_counts(patient_sets, data, code, col):
    counts = np.zeros((len(patient_sets), ))
    for i, set in enumerate(patient_sets):
        data_set = data.iloc[np.asarray(set)]
        counts[i] = data_set[data_set[col].apply(lambda x: code in x)].shape[0]
    return counts


def run_enrichment_for_code(code, col_name_all_data, col_name_data_to_test):
    counts_all_data_code = get_counts(patient_sets_all_data, codes_year, code, col_name_all_data)
    counts_data_to_test_code = get_counts(patient_sets_data_to_test, codes_hospital_year, code, col_name_data_to_test)
    if test == 't-test':
        stat, pval = ttest_ind(counts_all_data_code, counts_data_to_test_code)
    elif test == 'ks-test':
        stat, pval = ks_2samp(counts_all_data_code, counts_data_to_test_code)
    else:
        raise ValueError(f'Test {test} not implemented.')

    return pval, stat, counts_all_data_code, counts_data_to_test_code


for i in trange(np.max([len(all_diagnoses), len(all_chops)])):
    if len(all_chops) > i:
        pval_chop, stat_chop, counts_all_data_chop, counts_data_to_test_chop = run_enrichment_for_code(all_chops[i], col_name_all_data='all_procedures', col_name_data_to_test='all_procedures')
        chops_pvalues[i] = pval_chop
        chops_statistic[i] = stat_chop
        chops_mean_all[i] = np.mean(counts_all_data_chop)
        chops_mean_data_to_test[i] = np.mean(counts_data_to_test_chop)


    if len(all_diagnoses) > i:
        pval_diag, stat_diag, counts_all_data_diag, counts_data_to_test_diag = run_enrichment_for_code(all_diagnoses[i], col_name_all_data='all_diagnoses', col_name_data_to_test='all_diagnoses')
        diags_pvalues[i] = pval_diag
        diags_statistic[i] = stat_diag
        diags_mean_all[i] = np.mean(counts_all_data_diag)
        diags_mean_data_to_test[i] = np.mean(counts_data_to_test_diag)


_, diags_pvalues_adj, _, _ = multipletests(diags_pvalues, method='fdr_bh')
results_diags = pd.DataFrame({
    'diag': all_diagnoses,
    'pval': diags_pvalues,
    'stat': diags_statistic,
    'pval_adj_fdr': diags_pvalues_adj,
    'mean_all': diags_mean_all,
    f'mean_{hospital}_{year}': diags_mean_data_to_test
}).sort_values(by=['stat', 'pval'], ascending=[False, True])
results_diags.to_csv(os.path.join(dir_output, f'{test}_diagnoses.csv'), index=False)

_, chops_pvalues_adj, _, _ = multipletests(chops_pvalues, method='fdr_bh')
results_chops = pd.DataFrame({
    'chops': all_chops,
    'pval': chops_pvalues,
    'stat': chops_statistic,
    'pval_adj_fdr': chops_pvalues_adj,
    'mean_all': chops_mean_all,
    f'mean_{hospital}_{year}': chops_mean_data_to_test
}).sort_values(by=['stat', 'pval'], ascending=[False, True])
results_chops.to_csv(os.path.join(dir_output, f'{test}_chops.csv'), index=False)


print('')
