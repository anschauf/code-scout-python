import calendar
import re
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
from src.service.bfs_cases_db_service import get_all_revision_ids_containing_a_chop, \
    get_all_chops_for_revision_ids, get_codes, \
    get_sociodemographics_by_sociodemographics_ids, get_sociodemographics_for_year
from src.service.database import Database
from src.utils.global_configs import GROUPER_FORMAT_COL
from src.utils.group import format_for_grouper

# hopsital_name = 'Kantonsspital Winterthur'
discharge_year = 2020
replace = True

# dir_output = join(ROOT_DIR, 'results', 'missing_additional_chops_multilang', f'{hopsital_name}_{discharge_year}')
dir_output = join(ROOT_DIR, 'results', 'missing_additional_chops_multilang_reporting', f'{discharge_year}')
if not exists(dir_output):
    makedirs(dir_output)

filename_analysed_cases=join(dir_output, 'analysed_cases.csv')
if exists(filename_analysed_cases):
    analysed_cases = pd.read_csv(filename_analysed_cases)[SOCIODEMOGRAPHIC_ID_COL].tolist()
else:
    analysed_cases = list()

current_GMT = time.gmtime()
ts = calendar.timegm(current_GMT)
dt = datetime.fromtimestamp(ts)
dir_output_time_stamp = join(dir_output, str(dt).replace(' ', '_'))
if not exists(dir_output_time_stamp):
     makedirs(dir_output_time_stamp)

# chop_catalogue = wr.s3.read_csv('s3://aimedic-catalogues/chop/2020/CHOP 2020 Multilang CSV DE 2019_10_22.csv', sep=';')
chop_catalogue = wr.s3.read_csv('s3://aimedic-catalogues/chop/2021/CHOP 2021 Multilang CSV DE 2020_10_30.csv', sep=';')
text_field = 'DE'

# chop_catalogue = wr.s3.read_csv('s3://aimedic-catalogues/chop/2020/CHOP 2020_Systematisches_Verzeichnis_DE_2019_07_22.csv', sep=';')
# chop_catalogue = wr.s3.read_csv('s3://aimedic-catalogues/chop/2019/CHOP 2019_Systematisches_Verzeichnis_DE_2018_07_23.csv', sep=';', encoding='unicode_escape')
# text_field = 'text'

chop_catalogue['code'] = chop_catalogue['zcode'].apply(lambda zcode: zcode[1:])
chop_catalogue.sort_values(by='code', inplace=True) # sort to later get the codes in between a range of codes
chop_catalogue_additional_codes = chop_catalogue[chop_catalogue['item type'] == 'S']

regex_brackets = re.compile(r'\(\d.*?\)')
def extract_codes(text: str):
    all_matches = regex_brackets.findall(text)
    if len(all_matches) == 1:
        stripped_match = all_matches[0].replace('(','').replace(')','').split(', ')
        return [x.replace(' ', '') for x in stripped_match]
    elif len(all_matches) > 1:
        return [x.replace('(', '').replace(')', '').split(', ')[0].replace(' ', '') for x in all_matches]
    else:
        return []
chop_catalogue_additional_codes['extracted_additional_codes'] = chop_catalogue_additional_codes[text_field].apply(extract_codes)

def find(str, ch):
    for i, ltr in enumerate(str):
        if ltr == ch:
            yield i

def fill_up_ending_digits(codes: list[str]):
    all_codes = list()
    for code in codes:
        all_dashes = list(find(code, '-'))
        if code.endswith('-') and not len(all_dashes) > 1:
            extended_codes = [x for x in chop_catalogue['code'].values if x.startswith(code[:-1]) and len(x) > (len(code)-1)]
            all_codes.extend(list(np.unique(extended_codes)))
        else:
            all_codes.append(code)
    return all_codes
chop_catalogue_additional_codes['extracted_additional_codes'] = chop_catalogue_additional_codes['extracted_additional_codes'].apply(fill_up_ending_digits)

def fill_up_ranges(codes: list[str]):
    all_codes = list()
    for code in codes:
        all_dashes = list(find(code.replace(' ', ''), '-'))
        if any([True if 0 < dash < (len(code)-1) else False for dash in all_dashes]):
            # find range-dash in the middle (could also be that the first code has a dash at the end)

            # define lower and upper boundary codes
            if '--' in code:
                lower_boundary_code = code[:all_dashes[0]+1]
                upper_boundary_code = code[all_dashes[1]+1:]
            else:
                lower_boundary_code = code[:all_dashes[0]]
                upper_boundary_code = code[all_dashes[0]+1:]

            # replace dashes with full codes
            if lower_boundary_code.endswith('-'):
                all_lower_boundary_codes = list(np.unique([code for code in chop_catalogue['code'].values if code.startswith(lower_boundary_code[:-1])]))
                all_lower_boundary_codes.sort()
            else:
                all_lower_boundary_codes = [lower_boundary_code]
            if upper_boundary_code.endswith('-'):
                all_upper_boundary_codes = list(np.unique([code for code in chop_catalogue['code'].values if code.startswith(upper_boundary_code[:-1])]))
                all_upper_boundary_codes.sort()
            else:
                all_upper_boundary_codes = [upper_boundary_code]

            # get all codes in between the range based on a SORTED chop_catalogue DataFrame
            ind_lower_boundary = np.where(chop_catalogue['code'].values == all_lower_boundary_codes[0])[0]
            ind_upper_boundary = np.where(chop_catalogue['code'].values == all_upper_boundary_codes[-1])[0]
            all_codes_within_the_range = list(np.unique(chop_catalogue['code'].values[np.min(ind_lower_boundary):np.max(ind_upper_boundary)+1]))
            all_codes.extend(all_codes_within_the_range)
        else:
            all_codes.append(code)
    return all_codes

chop_catalogue_additional_codes['extracted_additional_codes'] = chop_catalogue_additional_codes['extracted_additional_codes'].apply(fill_up_ranges)
chop_catalogue_additional_codes = chop_catalogue_additional_codes[chop_catalogue_additional_codes['extracted_additional_codes'].apply(lambda l: len(l) > 0)]
chop_catalogue_additional_codes['code_without_dot'] = chop_catalogue_additional_codes['code'].apply(lambda code: code.replace('.', ''))
chop_catalogue_additional_codes['extracted_additional_codes_without_dot'] = chop_catalogue_additional_codes['extracted_additional_codes'].apply(lambda codes: [code.replace('.', '') for code in codes])

chop_catalogue_additional_codes.to_csv(join(dir_output_time_stamp, 'processed_chop_catalogue.csv'), index=False)
logger.info(f'Found {chop_catalogue_additional_codes.shape[0]} codes with additional codes.')
logger.info(f'All codes map in total to {chop_catalogue_additional_codes["extracted_additional_codes"].apply(lambda x: len(x)).sum()} additional codes.')


with Database() as db:
    all_code_appearances = get_all_revision_ids_containing_a_chop(chop_catalogue_additional_codes['code_without_dot'].tolist(), db.session)
    all_cases = get_sociodemographics_for_year(year=discharge_year, session=db.session)
    all_code_appearances = all_code_appearances[all_code_appearances[SOCIODEMOGRAPHIC_ID_COL].apply(lambda id: id in all_cases[SOCIODEMOGRAPHIC_ID_COL].values)]

    case_indices_missing_additional_chops = list()
    case_indices_replace_additional_chops = list()
    for row in tqdm(all_code_appearances.itertuples(), total=all_code_appearances.shape[0]):
        all_case_chops = get_all_chops_for_revision_ids([row.revision_id], db.session)

        # find chop in catalogue and check if additional code is missing
        additional_code_candidates = np.concatenate(chop_catalogue_additional_codes[chop_catalogue_additional_codes['code_without_dot'] == row.code]['extracted_additional_codes_without_dot'].values)
        has_additional_code = np.intersect1d(additional_code_candidates, all_case_chops['code'].values)
        if len(has_additional_code) == 0:
            case_indices_missing_additional_chops.append(row.Index)
        else:
            case_indices_replace_additional_chops.append(row.Index)

cases_missing_additional_chops = all_code_appearances.loc[case_indices_missing_additional_chops]
unique_missing_chop, count_missing_chop = np.unique(cases_missing_additional_chops['code'], return_counts=True)
missing_chops_summary = pd.DataFrame({
    'code': unique_missing_chop,
    'count': count_missing_chop
}).sort_values(by='count', ascending=False)

missing_chops_summary_catalogue = pd.merge(missing_chops_summary, chop_catalogue_additional_codes[['code_without_dot', 'extracted_additional_codes_without_dot']], left_on='code', right_on='code_without_dot', how='left')
missing_chops_summary_catalogue.drop(columns=['code_without_dot'], inplace=True)
missing_chops_summary_catalogue.rename(columns={'extracted_additional_codes_without_dot': 'extracted_additional_codes'}, inplace=True)
missing_chops_summary_catalogue.to_csv(join(dir_output_time_stamp, 'count_missing_chops.csv'), index=False)

cases_replace_additional_chops = all_code_appearances.loc[case_indices_replace_additional_chops]
unique_replace_chop, count_replace_chop = np.unique(cases_replace_additional_chops['code'], return_counts=True)
replace_chops_summary = pd.DataFrame({
    'code': unique_replace_chop,
    'count': count_replace_chop
}).sort_values(by='count', ascending=False)
replace_chops_summary_catalogue = pd.merge(replace_chops_summary, chop_catalogue_additional_codes[['code_without_dot', 'extracted_additional_codes_without_dot']], left_on='code', right_on='code_without_dot', how='left')
replace_chops_summary_catalogue.drop(columns=['code_without_dot'], inplace=True)
replace_chops_summary_catalogue.rename(columns={'extracted_additional_codes_without_dot': 'extracted_additional_codes'}, inplace=True)
replace_chops_summary_catalogue.to_csv(join(dir_output_time_stamp, 'count_replace_chops.csv'), index=False)


def get_original_case(original_codes, original_case_sociodemographics):
    original_case = pd.DataFrame([[case.sociodemographic_id,
                                   original_case_sociodemographics['case_id'].values[0],
                                   original_codes['old_pd'].values[0],
                                   original_codes['secondary_diagnoses'].values[0],
                                   original_codes['primary_procedure'].values[0],
                                   original_codes['secondary_procedures'].values[0],
                                   original_case_sociodemographics['gender'].values[0],
                                   original_case_sociodemographics['age_years'].values[0],
                                   original_case_sociodemographics['age_days'].values[0],
                                   original_case_sociodemographics['gestation_age'].values[0],
                                   original_case_sociodemographics['duration_of_stay'].values[0],
                                   original_case_sociodemographics['ventilation_hours'].values[0],
                                   original_case_sociodemographics['grouper_admission_type'].values[0],
                                   original_case_sociodemographics['admission_date'].values[0],
                                   original_case_sociodemographics['admission_weight'].values[0],
                                   original_case_sociodemographics['grouper_discharge_type'].values[0],
                                   original_case_sociodemographics['discharge_date'].values[0],
                                   original_case_sociodemographics['medications'].values[0]]],
                                 columns=['sociodemographic_id', 'case_id',
                                          'primary_diagnosis', 'secondary_diagnoses', 'primary_procedure',
                                          'secondary_procedures',
                                          'gender', 'age_years', 'age_days', 'gestation_age', 'duration_of_stay',
                                          'ventilation_hours',
                                          'grouper_admission_type', 'admission_date', 'admission_weight',
                                          'grouper_discharge_type', 'discharge_date', 'medications'])
    return original_case

def create_all_cases_with_add_ons(case, potential_missing_codes):
    # TODO double check concerning old_pd field
    # TODO deep copy of the original df did change the secondary procedures as well in the original case
    all_permuted_cases = list()
    for code in potential_missing_codes:
        original_codes = get_codes(pd.DataFrame({SOCIODEMOGRAPHIC_ID_COL: case.sociodemographic_id, 'revision_id': case.revision_id}, index=[0]), db.session)
        original_case_sociodemographics = get_sociodemographics_by_sociodemographics_ids([case.sociodemographic_id], db.session)
        permuted_case = get_original_case(original_codes, original_case_sociodemographics)
        all_chops_original_case = np.concatenate([original_codes['primary_procedure'].values, original_codes['secondary_procedures'].values[0]])
        ind_triggering_code = np.asarray([i for i in range(len(all_chops_original_case)) if all_chops_original_case[i].startswith(case.code)])
        code_to_add = all_chops_original_case[ind_triggering_code][0]
        code_to_add = code_to_add.replace(case.code, code)
        permuted_case['code_to_add'] = code_to_add
        permuted_case['secondary_procedures'].values[0].append(code_to_add)
        all_permuted_cases.append(permuted_case)

    original_codes = get_codes(pd.DataFrame({SOCIODEMOGRAPHIC_ID_COL: case.sociodemographic_id, 'revision_id': case.revision_id}, index=[0]),db.session)
    original_case_sociodemographics = get_sociodemographics_by_sociodemographics_ids([case.sociodemographic_id],db.session)
    return get_original_case(original_codes, original_case_sociodemographics), all_permuted_cases


def create_all_cases_with_replacements(case, potential_codes_for_replacement):
    original_codes = get_codes(pd.DataFrame({SOCIODEMOGRAPHIC_ID_COL: case.sociodemographic_id, 'revision_id': case.revision_id}, index=[0]), db.session)
    original_case_sociodemographics = get_sociodemographics_by_sociodemographics_ids([case.sociodemographic_id], db.session)

    secondary_procedures_to_replace = np.intersect1d(potential_codes_for_replacement, [[y.split(':')[0] for y in x] for x in original_codes['secondary_procedures'].values])
    # close_candidates_to_replace = [[x for x in potential_codes_for_replacement if x.startswith(c[:-1]) and x!=c] for c in secondary_procedures_to_replace]
    close_candidates_to_replace = [[x for x in potential_codes_for_replacement if x!=c] for c in secondary_procedures_to_replace]

    all_permuted_cases = list()
    for code_to_replace, all_candidates in zip(secondary_procedures_to_replace, close_candidates_to_replace):
        for candidate in all_candidates:
            original_codes = get_codes(pd.DataFrame({SOCIODEMOGRAPHIC_ID_COL: case.sociodemographic_id, 'revision_id': case.revision_id},index=[0]), db.session)
            original_case_sociodemographics = get_sociodemographics_by_sociodemographics_ids([case.sociodemographic_id],db.session)
            permuted_case = get_original_case(original_codes, original_case_sociodemographics)
            permuted_case['secondary_procedures'].values[0] = [x if not x.startswith(code_to_replace) else x.replace(code_to_replace, candidate) for x in permuted_case['secondary_procedures'].values[0]]
            permuted_case['code_to_replace'] = code_to_replace
            permuted_case['code_inserted'] = candidate
            all_permuted_cases.append(permuted_case)

    return get_original_case(original_codes, original_case_sociodemographics), all_permuted_cases



@dataclass(frozen=True)
class Case:
    case_id: str
    sociodemographic_id: str
    revision_id: str
    effective_cw: float
    code_to_add_on: str
    code_to_replace: list[str]
    codes_to_add: list[str]
    new_effective_cw: list[float]
    supplement_charges: list[float]


upcodeable_cases = list()
failed_cases = list()
def write_to_file():
    pd.DataFrame({
        'case_id': [x.case_id for x in upcodeable_cases],
        'sociodemographic_id': [x.sociodemographic_id for x in upcodeable_cases],
        'revision_id': [x.revision_id for x in upcodeable_cases],
        'effective_cw': [x.effective_cw for x in upcodeable_cases],
        'code_to_add_on': [x.code_to_add_on for x in upcodeable_cases],
        'code_to_replace': ['|'.join(x.code_to_replace) for x in upcodeable_cases],
        'codes_to_add': ['|'.join(x.codes_to_add) for x in upcodeable_cases],
        'new_effective_cw': ['|'.join([str(y) for y in x.new_effective_cw]) if len(x.new_effective_cw) > 1 else str(x.new_effective_cw[0]) for x in upcodeable_cases],
        'new_supplementary_charges': ['|'.join([str(y) for y in x.supplement_charges]) if len(x.supplement_charges) > 1 else str(x.supplement_charges[0]) for x in upcodeable_cases]
    }).to_csv(join(dir_output_time_stamp, 'upcodeable_cases.csv'), index=False)
    pd.DataFrame({'sociodemographic_id': [x.sociodemographic_id for x in failed_cases]}).to_csv(join(dir_output_time_stamp, 'failed_cases.csv'), index=False)
    if len(analysed_cases) > 1:
        pd.DataFrame({SOCIODEMOGRAPHIC_ID_COL: analysed_cases}).to_csv(filename_analysed_cases, index=False)

if replace:

    with Database() as db:
        for case in tqdm(cases_replace_additional_chops.itertuples(), total=cases_replace_additional_chops.shape[0]):
            if not case.sociodemographic_id in analysed_cases:
                potential_codes_for_replacement = chop_catalogue_additional_codes[chop_catalogue_additional_codes['code_without_dot'] == case.code]['extracted_additional_codes_without_dot'].values[0]
                original_case, all_permuted_cases = create_all_cases_with_replacements(case, potential_codes_for_replacement)
                try:

                    # group original case
                    original_case_formatted = format_for_grouper(original_case).iloc[0][GROUPER_FORMAT_COL]

                    # group all permuted cases
                    permuted_cases_formatted = [format_for_grouper(x).iloc[0][GROUPER_FORMAT_COL] for x in all_permuted_cases]
                    all_cases = [original_case_formatted] + permuted_cases_formatted
                    grouper_result = AIMEDIC_GROUPER.run_batch_grouper(cases=all_cases)
                    effective_cost_weight_original_case = grouper_result['effectiveCostWeight'].values[0]
                    supplement_charges_original_case = grouper_result['supplementCharges'].values[0]

                    list_code_to_replace = list()
                    list_code_inserted = list()
                    list_new_cw = list()
                    list_new_supplement_charges = list()
                    for i, upgrade in enumerate(grouper_result.itertuples()):
                        if np.logical_or(upgrade.effectiveCostWeight > effective_cost_weight_original_case,
                                upgrade.supplementCharges > supplement_charges_original_case) and not i == 0:
                            if (upgrade.effectiveCostWeight > effective_cost_weight_original_case):
                                logger.info(f'Higher CW based on {case.code}!')
                            else:
                                logger.info(f'Higher supplement charges based on {case.code}!')

                            code_to_replace = all_permuted_cases[i - 1]['code_to_replace'].values[0]
                            list_code_to_replace.append(code_to_replace)
                            code_replacement = all_permuted_cases[i - 1]['code_inserted'].values[0]
                            list_code_inserted.append(code_replacement)
                            list_new_cw.append(upgrade.effectiveCostWeight)
                            list_new_supplement_charges.append(upgrade.supplementCharges)


                    if len(list_code_to_replace) > 0:
                        logger.info(f'Found {len(list_code_to_replace)} suggestions for case sociodemographic_id {original_case[SOCIODEMOGRAPHIC_ID_COL].values[0]}')
                        upcodeable_cases.append(
                            Case(
                                case_id=original_case['case_id'].values[0],
                                sociodemographic_id=case.sociodemographic_id,
                                revision_id=case.revision_id,
                                effective_cw=grouper_result['effectiveCostWeight'].values[0],
                                code_to_add_on=case.code,
                                code_to_replace=list_code_to_replace,
                                codes_to_add=list_code_inserted,
                                new_effective_cw=list_new_cw,
                                supplement_charges=list_new_supplement_charges
                            )
                        )

                    analysed_cases.append(case.sociodemographic_id)

                except:
                    logger.warning(f'Case with sociodemographic_id: {original_case[SOCIODEMOGRAPHIC_ID_COL].values[0]} failed.')
                    failed_cases.append(case)

                finally:
                    write_to_file()

            else:
                logger.info(f'Case with sociodemographic_id {case.sociodemographic_id} already analysed.')



else:
    with Database() as db:
        for case in tqdm(cases_missing_additional_chops.itertuples(), total=cases_missing_additional_chops.shape[0]):
            if not case.sociodemographic_id in analysed_cases:
                potential_missing_codes = chop_catalogue_additional_codes[chop_catalogue_additional_codes['code_without_dot'] == case.code]['extracted_additional_codes_without_dot'].values[0]
                original_case, all_permuted_cases = create_all_cases_with_add_ons(case, potential_missing_codes)
                try:

                    # group original case
                    original_case_formatted = format_for_grouper(original_case).iloc[0][GROUPER_FORMAT_COL]

                    # group all permuted cases
                    permuted_cases_formatted = [format_for_grouper(x).iloc[0][GROUPER_FORMAT_COL] for x in all_permuted_cases]
                    all_cases = [original_case_formatted] + permuted_cases_formatted
                    grouper_result = AIMEDIC_GROUPER.run_batch_grouper(cases=all_cases)
                    effective_cost_weight_original_case = grouper_result['effectiveCostWeight'].values[0]
                    supplement_charges_original_case = grouper_result['supplementCharges'].values[0]


                    list_codes_upgrading = list()
                    list_new_cw = list()
                    list_new_supplement_charges = list()
                    for i, upgrade in enumerate(grouper_result.itertuples()):
                        if np.logical_or(upgrade.effectiveCostWeight > effective_cost_weight_original_case,
                                upgrade.supplementCharges > supplement_charges_original_case) and not i == 0:
                            if (upgrade.effectiveCostWeight > effective_cost_weight_original_case):
                                logger.info(f'Higher CW based on {case.code}!')
                            else:
                                logger.info(f'Higher supplement charges based on {case.code}!')

                            code_to_replace = all_permuted_cases[i - 1]['code_to_add'].values[0]
                            list_codes_upgrading.append(code_to_replace)
                            list_new_cw.append(upgrade.effectiveCostWeight)
                            list_new_supplement_charges.append(upgrade.supplementCharges)

                    if len(list_codes_upgrading) > 0:
                        logger.info(f'Found {len(list_codes_upgrading)} suggestions for case sociodemographic_id {original_case[SOCIODEMOGRAPHIC_ID_COL].values[0]}')
                        upcodeable_cases.append(
                            Case(
                                case_id=original_case['case_id'].values[0],
                                sociodemographic_id=case.sociodemographic_id,
                                revision_id=case.revision_id,
                                effective_cw=grouper_result['effectiveCostWeight'].values[0],
                                code_to_add_on=case.code,
                                codes_to_add=list_codes_upgrading,
                                code_to_replace=[''],
                                new_effective_cw=list_new_cw,
                                supplement_charges=list_new_supplement_charges
                            )
                        )

                    analysed_cases.append(case.sociodemographic_id)
                except:
                    logger.warning(f'Case with sociodemographic_id: {original_case[SOCIODEMOGRAPHIC_ID_COL].values[0]} failed.')
                    failed_cases.append(case)

                finally:
                    write_to_file()

            else:
                logger.info(f'Case with sociodemographic_id {case.sociodemographic_id} already analysed.')

print('')
