import calendar
import copy
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
from src.service.bfs_cases_db_service import get_all_revision_ids_containing_a_chop, get_sociodemographics_for_year, \
    get_codes, get_sociodemographics_by_sociodemographics_ids
from src.service.database import Database
from src.utils.global_configs import GROUPER_FORMAT_COL
from src.utils.group import format_for_grouper

discharge_year = 2019
# filename_chop_catalogue = 's3://aimedic-catalogues/chop/2020/CHOP 2020 Multilang CSV DE 2019_10_22_processed.csv'
filename_chop_catalogue = 's3://aimedic-catalogues/chop/2019/CHOP 2019 Multilang CSV DE 2018_10_30_processed.csv'

# dir_output = join(ROOT_DIR, 'results', 'missing_additional_chops_multilang', f'{hopsital_name}_{discharge_year}')
dir_output = join(ROOT_DIR, 'results', 'missing_additional_chops_multilang_reporting_new_implementation', f'{discharge_year}')
if not exists(dir_output):
    makedirs(dir_output)

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

chop_catalogue_additional_codes = wr.s3.read_csv(filename_chop_catalogue, sep=';')
chop_catalogue_additional_codes['extracted_additional_codes_without_dot'] = chop_catalogue_additional_codes['extracted_additional_codes_without_dot'].apply(lambda x: x.split('|'))


def get_original_case(original_case_sociodemographics):
    original_case = pd.DataFrame([[original_case_sociodemographics.sociodemographic_id,
                                   original_case_sociodemographics.case_id,
                                   original_case_sociodemographics.old_pd,
                                   original_case_sociodemographics.secondary_diagnoses,
                                   original_case_sociodemographics.primary_procedure,
                                   original_case_sociodemographics.secondary_procedures,
                                   original_case_sociodemographics.gender,
                                   original_case_sociodemographics.age_years,
                                   original_case_sociodemographics.age_days,
                                   original_case_sociodemographics.gestation_age,
                                   original_case_sociodemographics.duration_of_stay,
                                   original_case_sociodemographics.ventilation_hours,
                                   original_case_sociodemographics.grouper_admission_type,
                                   original_case_sociodemographics.admission_date,
                                   original_case_sociodemographics.admission_weight,
                                   original_case_sociodemographics.grouper_discharge_type,
                                   original_case_sociodemographics.discharge_date,
                                   original_case_sociodemographics.medications]],
                                 columns=['sociodemographic_id', 'case_id',
                                          'primary_diagnosis', 'secondary_diagnoses', 'primary_procedure',
                                          'secondary_procedures',
                                          'gender', 'age_years', 'age_days', 'gestation_age', 'duration_of_stay',
                                          'ventilation_hours',
                                          'grouper_admission_type', 'admission_date', 'admission_weight',
                                          'grouper_discharge_type', 'discharge_date', 'medications'])
    return original_case


def combine_chops(row):
    all_chops = list()
    if isinstance(row.primary_procedure, str):
        all_chops.append(row.primary_procedure.split(':')[0])
    if isinstance(row.secondary_procedures, list):
        all_chops.extend([x.split(':')[0] for x in row.secondary_procedures])
    return all_chops


@dataclass(frozen=True)
class Case:
    sociodemographic_id: str
    effective_cw: float
    supplement_charges: float
    drg: str
    triggering_chop: str
    chop_to_replace: str
    chop_to_insert: str
    new_effective_cw: float
    new_supplement_charges: float
    new_drg: str


def write_to_file():
    pd.DataFrame({
        'sociodemographic_id': [x.sociodemographic_id for x in list_upcoded_cases],
        'effective_cw': [x.effective_cw for x in list_upcoded_cases],
        'supplement_charges': [x.supplement_charges for x in list_upcoded_cases],
        'drg': [x.drg for x in list_upcoded_cases],
        'triggering_chop': [x.triggering_chop for x in list_upcoded_cases],
        'chop_to_replace': [x.chop_to_replace for x in list_upcoded_cases],
        'chop_to_insert': [x.chop_to_insert for x in list_upcoded_cases],
        'new_effective_cw': [x.new_effective_cw for x in list_upcoded_cases],
        'new_supplement_charges': [x.new_supplement_charges for x in list_upcoded_cases],
        'new_drg': [x.new_drg for x in list_upcoded_cases]
    }).to_csv(join(dir_output_time_stamp, 'upcodeable_codes.csv'), index=False)
    if len(analysed_codes) > 1:
        pd.DataFrame({'enriched_codes': analysed_codes}).to_csv(filename_analysed_codes, index=False)


list_upcoded_cases = list()
with Database() as db:
    all_cases_of_discharge_year = get_sociodemographics_for_year(year=discharge_year, session=db.session)
    for chop_with_additional_codes in tqdm(chop_catalogue_additional_codes.itertuples(), total=chop_catalogue_additional_codes.shape[0]):
        logger.info(f'Looking for cases to upgrade containing CHOP {chop_with_additional_codes.code_without_dot}')

        try:
            # find all cases which contain the current chop of interest, where we look to replace one of the "ebenso kodieren" chops with another "ebenso kodieren" chop
            all_revision_ids_containing_chop = get_all_revision_ids_containing_a_chop([chop_with_additional_codes.code_without_dot], db.session)
            if all_revision_ids_containing_chop.shape[0] > 0:

                # filter for cases in discharge year
                all_revision_ids_containing_chop_from_discharge_year = all_revision_ids_containing_chop[all_revision_ids_containing_chop[SOCIODEMOGRAPHIC_ID_COL].isin(all_cases_of_discharge_year[SOCIODEMOGRAPHIC_ID_COL])]
                if all_revision_ids_containing_chop_from_discharge_year.shape[0] > 0:

                    # get codes for revision ide
                    all_case_chops = get_codes(all_revision_ids_containing_chop_from_discharge_year[[SOCIODEMOGRAPHIC_ID_COL, 'revision_id']], db.session)
                    all_case_chops['all_chops'] = all_case_chops.apply(combine_chops, axis=1)
                    all_case_chops.drop_duplicates(subset=[SOCIODEMOGRAPHIC_ID_COL, 'revision_id'], inplace=True)

                    # filter all cases whether they contain an "ebenso kodieren" chop
                    all_cases_containing_chop_and_additional_chop = all_case_chops[all_case_chops['all_chops'].apply(lambda x: len(np.intersect1d(x, chop_with_additional_codes.extracted_additional_codes_without_dot)) > 0)]
                    if all_cases_containing_chop_and_additional_chop.shape[0] > 0:
                        all_socios_with_chop = get_sociodemographics_by_sociodemographics_ids(all_cases_containing_chop_and_additional_chop[SOCIODEMOGRAPHIC_ID_COL].tolist(), db.session)
                        all_case_infos = pd.merge(all_cases_containing_chop_and_additional_chop, all_socios_with_chop, on=[SOCIODEMOGRAPHIC_ID_COL], how='left')

                        # get grouper results for original cases
                        original_cases = [format_for_grouper(get_original_case(x)).iloc[0][GROUPER_FORMAT_COL] for x in all_case_infos.itertuples()]
                        grouper_result_original_cases = AIMEDIC_GROUPER.run_batch_grouper(cases=original_cases)

                        # replace chop with code in "ebenso kodieren"
                        # all combinations are possible, only the triggering chop for "ebenso kodieren" should be present in all of them
                        for chop_to_replace in chop_with_additional_codes.extracted_additional_codes_without_dot:
                            for chop_to_insert in chop_with_additional_codes.extracted_additional_codes_without_dot:
                                if chop_to_replace != chop_to_insert:
                                    permuted_cases = copy.deepcopy(original_cases)
                                    permuted_cases = [x.replace(chop_to_replace+':', chop_to_insert+':') for x in permuted_cases]
                                    grouper_result_permuted_cases = AIMEDIC_GROUPER.run_batch_grouper(cases=permuted_cases)

                                    grouper_result_comparison = pd.merge(grouper_result_original_cases, grouper_result_permuted_cases, on=['id', 'entryDate', 'exitDate'], how='right', suffixes=('_original', '_permuted'))
                                    ind_cases_higher_cw = np.where(np.logical_or(
                                        grouper_result_comparison['effectiveCostWeight_permuted'].values - grouper_result_comparison['effectiveCostWeight_original'].values > 0,
                                        grouper_result_comparison['supplementCharges_permuted'].values - grouper_result_comparison['supplementCharges_original'].values > 0))[0]

                                    if len(ind_cases_higher_cw) > 0:
                                        for ind in ind_cases_higher_cw:
                                            logger.success(f'Found code replacement {chop_to_insert} for {chop_to_replace} triggering in case {grouper_result_comparison["id"].values[ind]}')

                                            list_upcoded_cases.append(Case(
                                                sociodemographic_id=grouper_result_comparison['id'].values[ind],
                                                effective_cw=grouper_result_comparison['effectiveCostWeight_original'].values[ind],
                                                supplement_charges=grouper_result_comparison['supplementCharges_original'].values[ind],
                                                drg=grouper_result_comparison['drg_original'].values[ind],
                                                triggering_chop=chop_with_additional_codes.code_without_dot,
                                                chop_to_replace=chop_to_replace,
                                                chop_to_insert=chop_to_insert,
                                                new_effective_cw=grouper_result_comparison['effectiveCostWeight_permuted'].values[ind],
                                                new_supplement_charges=grouper_result_comparison['supplementCharges_permuted'].values[ind],
                                                new_drg=grouper_result_comparison['drg_permuted'].values[ind]
                                            ))
        except BaseException as e:
            logger.error('Failed to do something: ' + str(e))
            logger.warning(f'Enriched code {chop_with_additional_codes.code_without_dot} failed.')

        finally:
            analysed_codes.append(chop_with_additional_codes.code_without_dot)
            write_to_file()

print('')
