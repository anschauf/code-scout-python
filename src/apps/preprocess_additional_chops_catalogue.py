import re

import awswrangler as wr
import numpy as np
from loguru import logger

discharge_year = 2020


def extract_codes(text: str):
    regex_brackets = re.compile(r'\(\d.*?\)')
    all_matches = regex_brackets.findall(text)
    if len(all_matches) == 1:
        stripped_match = all_matches[0].replace('(','').replace(')','').split(', ')
        return [x.replace(' ', '') for x in stripped_match]
    elif len(all_matches) > 1:
        return [x.replace('(', '').replace(')', '').split(', ')[0].replace(' ', '') for x in all_matches]
    else:
        return []


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


if __name__ == '__main__':

    text_field = 'DE'
    for filename_catelogue in [
        's3://aimedic-catalogues/chop/2022/CHOP 2022 Multilang CSV DE 2021_10_22.csv',
        's3://aimedic-catalogues/chop/2019/CHOP 2019 Multilang CSV DE 2018_10_30.csv',
        's3://aimedic-catalogues/chop/2020/CHOP 2020 Multilang CSV DE 2019_10_22.csv',
        's3://aimedic-catalogues/chop/2021/CHOP 2021 Multilang CSV DE 2020_10_30.csv'
    ]:
        logger.info(f'Preprocessing {filename_catelogue}')

        chop_catalogue = wr.s3.read_csv(filename_catelogue, sep=';')
        chop_catalogue['code'] = chop_catalogue['zcode'].apply(lambda zcode: zcode[1:])
        chop_catalogue.sort_values(by='code', inplace=True) # sort to later get the codes in between a range of codes
        chop_catalogue_additional_codes = chop_catalogue[chop_catalogue['item type'] == 'S']

        chop_catalogue_additional_codes['extracted_additional_codes'] = chop_catalogue_additional_codes[text_field].apply(extract_codes)
        chop_catalogue_additional_codes['extracted_additional_codes'] = chop_catalogue_additional_codes['extracted_additional_codes'].apply(fill_up_ending_digits)
        chop_catalogue_additional_codes['extracted_additional_codes'] = chop_catalogue_additional_codes['extracted_additional_codes'].apply(fill_up_ranges)
        chop_catalogue_additional_codes = chop_catalogue_additional_codes[chop_catalogue_additional_codes['extracted_additional_codes'].apply(lambda l: len(l) > 0)]
        chop_catalogue_additional_codes['code_without_dot'] = chop_catalogue_additional_codes['code'].apply(lambda code: code.replace('.', ''))
        chop_catalogue_additional_codes['extracted_additional_codes_without_dot'] = chop_catalogue_additional_codes['extracted_additional_codes'].apply(lambda codes: [code.replace('.', '') for code in codes])

        wr.s3.to_csv(chop_catalogue_additional_codes, filename_catelogue.replace('.csv', '_processed.csv'), index=False)
        logger.success(f'Found {chop_catalogue_additional_codes.shape[0]} codes with additional codes.')
        logger.success(f'All codes map in total to {chop_catalogue_additional_codes["extracted_additional_codes"].apply(lambda x: len(x)).sum()} additional codes.')
