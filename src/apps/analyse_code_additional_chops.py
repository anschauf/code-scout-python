import re

import awswrangler as wr
import numpy as np
from tqdm import tqdm

from src.models.sociodemographics import SOCIODEMOGRAPHIC_ID_COL
from src.service.bfs_cases_db_service import get_all_revision_ids_containing_a_chop, \
    get_sociodemographics_for_hospital_year, get_all_chops_for_revision_ids
from src.service.database import Database

hopsital_name = 'Kantonsspital Winterthur'
discharge_year = 2020

chop_catalogue = wr.s3.read_csv('s3://aimedic-catalogues/chop/2020/CHOP 2020_Systematisches_Verzeichnis_DE_2019_07_22.csv', sep=';')
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
chop_catalogue_additional_codes['extracted_additional_codes'] = chop_catalogue_additional_codes['text'].apply(extract_codes)

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


with Database() as db:
    all_code_appearances = get_all_revision_ids_containing_a_chop(chop_catalogue_additional_codes['code_without_dot'].tolist(), db.session)
    all_cases = get_sociodemographics_for_hospital_year(hospital_name=hopsital_name, year=discharge_year, session=db.session)
    all_code_appearances = all_code_appearances[all_code_appearances[SOCIODEMOGRAPHIC_ID_COL].apply(lambda id: id in all_cases[SOCIODEMOGRAPHIC_ID_COL].values)]

    case_indices_missing_additional_chops = list()
    for row in tqdm(all_code_appearances.itertuples(), total=all_code_appearances.shape[0]):
        all_case_chops = get_all_chops_for_revision_ids([row.revision_id], db.session)

        # find chop in catalogue and check if additional code is missing
        additional_code_candidates = np.concatenate(chop_catalogue_additional_codes[chop_catalogue_additional_codes['code_without_dot'] == row.code]['extracted_additional_codes_without_dot'].values)
        has_additional_code = np.intersect1d(additional_code_candidates, all_case_chops['code'].values)
        if len(has_additional_code) == 0:
            case_indices_missing_additional_chops.append(row.Index)

cases_missing_additional_chops = all_code_appearances.loc[case_indices_missing_additional_chops]



print('')
