import calendar
import copy
import os.path
import time
from dataclasses import dataclass
from datetime import datetime
from os import makedirs
from os.path import join, exists

import awswrangler as wr
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from src import ROOT_DIR
from src.models.sociodemographics import SOCIODEMOGRAPHIC_ID_COL
from src.service.aimedic_grouper import AIMEDIC_GROUPER
from src.service.bfs_cases_db_service import get_all_unique_sociodemographic_ids_with_procedures, \
    get_sociodemographics_by_sociodemographics_ids, get_codes, get_original_revision_id_for_sociodemographic_ids
from src.service.database import Database
from src.utils.global_configs import GROUPER_FORMAT_COL
from src.utils.group import format_for_grouper

hospital_id = 29
discharge_year = 2019


filename_enrichment = join(ROOT_DIR, 'results/code_enrichment_analysis/Kantonsspital Winterthur_2019/t-test_chops.csv')
dir_output = os.path.dirname(filename_enrichment)

filename_analysed_codes=join(dir_output, 'analysed_codes.csv')
if exists(filename_analysed_codes):
    analysed_codes = pd.read_csv(filename_analysed_codes)['enriched_codes'].tolist()
else:
    analysed_codes = list()

current_GMT = time.gmtime()
ts = calendar.timegm(current_GMT)
dt = datetime.fromtimestamp(ts)
dir_output_time_stamp = join(dir_output, str(dt).replace(' ', '_'))
if not exists(dir_output_time_stamp):
     makedirs(dir_output_time_stamp)

enrichment_results = pd.read_csv(filename_enrichment)
enrichment_results_significant = enrichment_results[enrichment_results['pval_adj_fdr'] < 0.05]
enrichment_results_significant.sort_values(by='pval_adj_fdr', ascending=True, inplace=True)

chop_catalogue = wr.s3.read_csv('s3://aimedic-catalogues/chop/2019/CHOP 2019_Systematisches_Verzeichnis_DE_2018_07_23.csv', sep=';', encoding='unicode_escape')
chop_catalogue['code'] = chop_catalogue['zcode'].apply(lambda x: x[1:].replace('.',''))


def get_similar_codes(code, n_variable_digits=1):
    return np.unique([x for x in chop_catalogue['code'] if x.startswith(code[:-n_variable_digits]) and x!=code]).tolist()


@dataclass(frozen=True)
class Case:
    sociodemographic_id: str
    effective_cw: float
    supplement_charges: float
    drg: str
    enriched_code: str
    pval: float
    statistic: float
    mean_all: float
    mean_test_hospital: float
    codes_to_add: str
    new_effective_cw: float
    new_supplement_charges: float
    new_drg: str


# def get_original_case(original_codes, original_case_sociodemographics):
def get_original_case(original_case):
    original_case = pd.DataFrame([[original_case.sociodemographic_id,
                                   original_case.case_id,
                                   original_case.old_pd,
                                   original_case.secondary_diagnoses,
                                   original_case.primary_procedure,
                                   original_case.secondary_procedures,
                                   original_case.gender,
                                   original_case.age_years,
                                   original_case.age_days,
                                   original_case.gestation_age,
                                   original_case.duration_of_stay,
                                   original_case.ventilation_hours,
                                   original_case.grouper_admission_type,
                                   original_case.admission_date,
                                   original_case.admission_weight,
                                   original_case.grouper_discharge_type,
                                   original_case.discharge_date,
                                   original_case.medications]],
                                 columns=['sociodemographic_id', 'case_id',
                                          'primary_diagnosis', 'secondary_diagnoses', 'primary_procedure',
                                          'secondary_procedures',
                                          'gender', 'age_years', 'age_days', 'gestation_age', 'duration_of_stay',
                                          'ventilation_hours',
                                          'grouper_admission_type', 'admission_date', 'admission_weight',
                                          'grouper_discharge_type', 'discharge_date', 'medications'])
    return original_case


def replace_procedure(original_case, procedure_to_replace, new_procedure):
    new_case = original_case.copy(deep=True)
    new_case['primary_procedure'].values[0] = original_case['primary_procedure'].values[0].replace(procedure_to_replace, new_procedure)
    new_case['secondary_procedures'].values[0] = [x.replace(procedure_to_replace, new_procedure) for x in original_case['secondary_procedures'].values[0]]
    new_case['added_procedure'] = new_procedure
    return new_case


list_upcoded_cases = list()


def write_to_file():
    pd.DataFrame({
        'sociodemographic_id': [x.sociodemographic_id for x in list_upcoded_cases],
        'effective_cw': [x.effective_cw for x in list_upcoded_cases],
        'supplement_charges': [x.supplement_charges for x in list_upcoded_cases],
        'drg': [x.drg for x in list_upcoded_cases],
        'enriched_code': [x.enriched_code for x in list_upcoded_cases],
        'pval': [x.pval for x in list_upcoded_cases],
        'statistic': [x.statistic for x in list_upcoded_cases],
        'mean_all': [x.mean_all for x in list_upcoded_cases],
        'mean_test_hospital': [x.mean_test_hospital for x in list_upcoded_cases],
        'codes_to_add': [x.codes_to_add for x in list_upcoded_cases],
        'new_effective_cw': [x.new_effective_cw for x in list_upcoded_cases],
        'new_supplement_charges': [x.new_supplement_charges for x in list_upcoded_cases],
        'new_drg': [x.new_drg for x in list_upcoded_cases]
    }).to_csv(join(dir_output_time_stamp, 'upcodeable_codes.csv'), index=False)
    if len(analysed_codes) > 1:
        pd.DataFrame({'enriched_codes': analysed_codes}).to_csv(filename_analysed_codes, index=False)



with Database() as db:
    for enriched_code in tqdm(enrichment_results_significant.itertuples(), total=enrichment_results_significant.shape[0]):
        if not enriched_code.chops in analysed_codes:
            similar_codes = get_similar_codes(enriched_code.chops)
            logger.info(f'Found {len(similar_codes)} similar codes for {enriched_code.chops}')
            if len(similar_codes) > 0:
                try:
                    all_socio_ids_with_chop = get_all_unique_sociodemographic_ids_with_procedures(enriched_code.chops, db.session)
                    all_socios_with_chop = get_sociodemographics_by_sociodemographics_ids(all_socio_ids_with_chop[SOCIODEMOGRAPHIC_ID_COL].tolist(), db.session)
                    all_socios_with_chop = all_socios_with_chop[(all_socios_with_chop['discharge_year'] == discharge_year) & (all_socios_with_chop['hospital_id'] == hospital_id)]


                    revision_ids = get_original_revision_id_for_sociodemographic_ids(all_socios_with_chop[SOCIODEMOGRAPHIC_ID_COL].tolist(), db.session)
                    all_socios_with_chop_revision_id = pd.merge(all_socios_with_chop, revision_ids, on=SOCIODEMOGRAPHIC_ID_COL, how='right')
                    all_socios_with_chop_revision_id_codes = get_codes(all_socios_with_chop_revision_id, db.session)
                    original_cases = [format_for_grouper(get_original_case(x)).iloc[0][GROUPER_FORMAT_COL] for x in all_socios_with_chop_revision_id_codes.itertuples()]
                    grouper_result_original_cases = AIMEDIC_GROUPER.run_batch_grouper(cases=original_cases)

                    logger.info(f'Found {len(original_cases)} cases containing {enriched_code.chops}')
                    for code_to_add in similar_codes:
                        permuted_cases = copy.deepcopy(original_cases)
                        permuted_cases = [x.replace(enriched_code.chops+':', code_to_add+':') for x in original_cases]
                        grouper_result_permuted_cases = AIMEDIC_GROUPER.run_batch_grouper(cases=permuted_cases)

                        if grouper_result_permuted_cases.shape[0] > 0:
                            grouper_result_permuted_cases = grouper_result_permuted_cases[grouper_result_permuted_cases['drgRelevantProcedures'].apply(lambda x: code_to_add in x)]
                            grouper_result_comparison = pd.merge(grouper_result_original_cases, grouper_result_permuted_cases, on=['id', 'entryDate', 'exitDate'], how='right', suffixes=('_original', '_permuted'))

                            if grouper_result_permuted_cases.shape[0] > 0:
                                ind_cases_higher_cw = np.where(np.logical_or(grouper_result_comparison['effectiveCostWeight_permuted'].values - grouper_result_comparison['effectiveCostWeight_original'].values > 0,
                                                                             grouper_result_comparison['supplementCharges_permuted'].values - grouper_result_comparison['supplementCharges_original'].values > 0))[0]

                                if len(ind_cases_higher_cw) > 0:
                                    for ind in ind_cases_higher_cw:
                                        logger.success(f'Found code replacement {code_to_add} for {enriched_code.chops} triggering in case {grouper_result_comparison["id"].values[ind]}')

                                        list_upcoded_cases.append(Case(
                                            sociodemographic_id=grouper_result_comparison['id'].values[ind],
                                            effective_cw=grouper_result_comparison['effectiveCostWeight_original'].values[ind],
                                            supplement_charges=grouper_result_comparison['supplementCharges_original'].values[ind],
                                            drg=grouper_result_comparison['drg_original'].values[ind],
                                            enriched_code=enriched_code.chops,
                                            pval=enriched_code.pval_adj_fdr,
                                            statistic=enriched_code.stat,
                                            mean_all=enriched_code.mean_all,
                                            mean_test_hospital=enriched_code.mean_test,
                                            codes_to_add=code_to_add,
                                            new_effective_cw=grouper_result_comparison['effectiveCostWeight_permuted'].values[ind],
                                            new_supplement_charges=grouper_result_comparison['supplementCharges_permuted'].values[ind],
                                            new_drg=grouper_result_comparison['drg_permuted'].values[ind]
                                        ))

                except BaseException as e:
                    logger.error('Failed to do something: ' + str(e))
                    logger.warning(f'Enriched code {enriched_code.chops} failed.')

                finally:
                    analysed_codes.append(enriched_code.chops)
                    write_to_file()

        else:
            logger.info(f'Enriched CHOP {enriched_code.chops} already analysed.')

print('')
