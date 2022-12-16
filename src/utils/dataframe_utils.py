import pandas as pd
from beartype import beartype
from loguru import logger

from src.utils.chop_validation import split_chop_codes, validate_chop_codes_list
from src.utils.icd_validation import validate_icd_codes_list


@beartype
def remove_duplicated_chops(df: pd.DataFrame,
                            *,
                            added_chops_col: str = 'added_chops',
                            removed_chops_col: str = 'removed_chops',
                            cleaned_added_chops_col: str = 'cleaned_added_chops',
                            cleaned_removed_chops_col: str = 'cleaned_removed_chops',
                            ) -> pd.DataFrame:
    """Compare 2 lists of CHOP codes, which are formatted as '<code>:<side>:<date>', and remove the codes which appear
    in both lists, regardless of their casing.

    @param df: The data where to perform the filter.
    @param added_chops_col: The column containing the added CHOPs.
    @param removed_chops_col: The column containing the removed CHOPs.
    @param cleaned_added_chops_col: The output column where to store the added CHOPs.
    @param cleaned_removed_chops_col: The output column where to store the removed CHOPs.
    @return: The input DataFrame, with the columns `cleaned_added_chops_col` and `cleaned_removed_chops_col` added,
        possibly overwriting existing columns.
    """
    def _remove_duplicated_chops(row):
        row[cleaned_added_chops_col], row[cleaned_removed_chops_col] = _remove_duplicates_case_insensitive(row[added_chops_col], row[removed_chops_col])
        return row

    df = df.apply(_remove_duplicated_chops, axis=1)

    return df


@beartype
def validate_icd_codes(df: pd.DataFrame,
                       *,
                       icd_codes_col: str = 'added_icds',
                       output_icd_codes_col: str = 'added_icds',
                       ) -> pd.DataFrame:
    """Validate whether a list of supposed ICD codes is actually made of ICD codes, discarding those which don't fit the
    known pattern for ICD codes.

    @param df: The data where to perform the filter.
    @param icd_codes_col: The column containing the ICDs to validate.
    @param output_icd_codes_col: The column where to store the results of the filtering / validation.
    @return: The input DataFrame, with the column `output_icd_codes_col` added, possibly overwriting an existing column.
    """
    invalid_rows_info = list()

    def _validate_icd_codes(row):
        original_codes = _filter_empty_strings(row[icd_codes_col])
        result = validate_icd_codes_list(original_codes)

        original_codes_for_comparison = [icd.replace('.', '').upper() for icd in original_codes]
        different_codes = _get_list_diff_case_insensitive(original_codes_for_comparison, result)
        if len(different_codes) > 0:
            invalid_rows_info.append(f'row {row.name}: discarded ICDs after validation {different_codes}')

        row[output_icd_codes_col] = result
        return row

    df = df.apply(_validate_icd_codes, axis=1)

    if len(invalid_rows_info) > 0:
        log_message = '\n'.join(invalid_rows_info)
        logger.info(f"Validated ICD codes in '{icd_codes_col}' and stored them into '{output_icd_codes_col}': The following {len(invalid_rows_info)} rows were affected:\n{log_message}")

    return df


@beartype
def validate_chop_codes(df: pd.DataFrame,
                        *,
                        chop_codes_col: str = 'added_chops',
                        output_chop_codes_col: str = 'added_chops',
                        ) -> pd.DataFrame:
    """Validate whether a list of supposed CHOP codes is actually made of CHOP codes, discarding those which don't fit
    the known pattern for CHOP codes.

    @param df: The data where to perform the filter.
    @param chop_codes_col: The column containing the CHOPs to validate.
    @param output_chop_codes_col: The column where to store the results of the filtering / validation.
    @return: The input DataFrame, with the column `output_chop_codes_col` added, possibly overwriting an existing column.
    """
    invalid_rows_info = list()

    def _validate_chop_codes(row):
        original_codes = _filter_empty_strings(row[chop_codes_col])
        result = validate_chop_codes_list(original_codes)

        different_codes = _get_list_diff_case_insensitive(original_codes, result, codes_are_chops=True)
        if len(different_codes) > 0:
            invalid_rows_info.append(f'row {row.name}: discarded CHOPs after validation {different_codes}')

        row[output_chop_codes_col] = result
        return row

    df = df.apply(_validate_chop_codes, axis=1)

    if len(invalid_rows_info) > 0:
        log_message = '\n'.join(invalid_rows_info)
        logger.info(f"Validated CHOP codes in '{chop_codes_col}' and stored them into '{output_chop_codes_col}': The following {len(invalid_rows_info)} rows were affected:\n{log_message}")

    return df


@beartype
def validate_pd_revised_sd(df: pd.DataFrame,
                           *,
                           pd_col: str = 'old_pd',
                           pd_new_col: str = 'new_pd',
                           added_icd_col: str = 'added_icds',
                           removed_icd_col: str = 'removed_icds',
                           ) -> pd.DataFrame:
    """Compare lists of old and new primary diagnoses, discarding codes which are not correctly captured. "new_pd" is
    deleted from "added_icds" if it appears in "added_icds". "old_pd" is removed from "removed_icds" if it appears in
    "removed_icds".

    @param df: The data where to perform the filter.
    @param pd_col: The column containing the unrevised primary diagnosis to validate.
    @param pd_new_col: The column containing the revised primary diagnosis to validate.
    @param added_icd_col: The column containing the added secondary icds to validate.
    @param removed_icd_col: The column containing the removed secondary icds to validate.
    @return: The input DataFrame, with a validated 'pd' (consolidated column with 'pd' and 'pd_new'), 'added_icds',
        'removed_icds', overwriting existing columns
    """
    def _validate_pd_revised_sd(row):
        old_pd = row[pd_col]
        new_pd = row[pd_new_col]
        added_icds = row[added_icd_col]
        removed_icds = row[removed_icd_col]

        if old_pd != new_pd:
            if new_pd in added_icds:
                added_icds.remove(new_pd)

            if old_pd in removed_icds:
                removed_icds.remove(old_pd)

        return row

    df = df.apply(_validate_pd_revised_sd, axis=1)
    return df


@beartype
def _remove_duplicates_case_insensitive(codes_list1: list[str], codes_list2: list[str]) -> tuple[list[str], list[str]]:
    """Compare 2 lists of CHOP codes, which are formatted as '<code>:<side>:<date>', and remove the codes which appear
    in both lists, regardless of their casing.

    @param codes_list1: The first list.
    @param codes_list2: The second list.
    @return: A tuple of (cleaned list 1, cleaned list 2), which do not contain the codes which appear in both.

    @note: If the intersection is empty, the original lists are returned, without being copied.
    """
    codes_list1_split = split_chop_codes(codes_list1)
    codes_list2_split = split_chop_codes(codes_list2)

    # Make a set out of the codes in each list, and convert them all to upper-case
    code_set1 = {codes_info[0].upper() for codes_info in codes_list1_split}
    code_set2 = {codes_info[0].upper() for codes_info in codes_list2_split}
    duplicates = code_set1.intersection(code_set2)

    if len(duplicates) == 0:
        return codes_list1, codes_list2

    else:
        cleaned_codes_list1_split = _filter_out_codes_from_list(codes_list1_split, codes_to_filter_out=duplicates)
        cleaned_codes_list2_split = _filter_out_codes_from_list(codes_list2_split, codes_to_filter_out=duplicates)

        # Re-concatenate the info on each CHOP into a single string, separated by a ":"
        cleaned_codes_list1 = [':'.join(codes_info) for codes_info in cleaned_codes_list1_split]
        cleaned_codes_list2 = [':'.join(codes_info) for codes_info in cleaned_codes_list2_split]
        return cleaned_codes_list1, cleaned_codes_list2


@beartype
def _filter_out_codes_from_list(codes_list: list[list[str]], *, codes_to_filter_out: set[str]) -> list[list[str]]:
    """Remove codes from a list of CHOP codes info. The codes are compared all upper-cased.

    @param codes_list: The list of codes to check.
    @param codes_to_filter_out: The list of codes to remove from the `codes_list`.
    @return: The filtered list of codes from `codes_list`.
    """
    clean_codes_list = list()
    for codes_info in codes_list:
        if codes_info[0].upper() not in codes_to_filter_out:
            clean_codes_list.append(codes_info)

    return clean_codes_list


@beartype
def _filter_empty_strings(lst: list[str]) -> list[str]:
    """Remove empty strings from a list."""
    return [item for item in lst if item != '']


@beartype
def _get_list_diff_case_insensitive(list1: list[str], list2: list[str], *, codes_are_chops: bool = False) -> set[str]:
    """Lists the items which are different between 2 lists, returning them upper-case."""
    def do_fix_list(lst: list, *, add_missing_dots: bool = False) -> set:
        a_set = set()
        for item in lst:
            fixed_item = item.strip().upper()

            if codes_are_chops:
                item_parts = fixed_item.split(':')
                if len(item_parts) < 3:
                    all_parts = item_parts + [''] * (3 - len(item_parts))
                    fixed_item = ':'.join(all_parts)

            a_set.add(fixed_item)

        return a_set

    upper_list1 = do_fix_list(list1)
    upper_list2 = do_fix_list(list2)

    different_items = set(upper_list1).difference(set(upper_list2))
    return different_items
