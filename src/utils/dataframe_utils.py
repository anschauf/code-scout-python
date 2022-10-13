import pandas as pd
from beartype import beartype
from loguru import logger

from src.utils.chop_validation import validate_chop_codes_list
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
    def _validate_icd_codes(row):
        result = validate_icd_codes_list(row[icd_codes_col])

        # Log changes
        different_codes = set(row[icd_codes_col]).difference(set(result))
        if len(different_codes) > 0:
            logger.debug(f"row {row.name}: discarded ICDs after validation {different_codes}")

        row[output_icd_codes_col] = result
        return row

    df = df.apply(_validate_icd_codes, axis=1)
    return df


@beartype
def validate_chop_codes(df: pd.DataFrame,
                        *,
                        chop_codes_col: str = 'added_chops',
                        chop_codes_deleted_col: str = 'removed_chops',
                        output_chop_codes_col: str = 'added_chops',
                        ) -> pd.DataFrame:
    """Validate whether a list of supposed CHOP codes is actually made of CHOP codes, discarding those which don't fit
    the known pattern for CHOP codes.

    @param df: The data where to perform the filter.
    @param chop_codes_col: The column containing the CHOPs to validate.
    @param output_chop_codes_col: The column where to store the results of the filtering / validation.
    @return: The input DataFrame, with the column `output_chop_codes_col` added, possibly overwriting an existing column.
    """
    def _validate_chop_codes(row):
        valid_chop_codes, invalid_chop_codes = validate_chop_codes_list(row[chop_codes_col])
        valid_chop_codes_deleted, invalid_chop_codes_deleted = validate_chop_codes_list(row[chop_codes_deleted_col])

        # Log changes
        # different_valid_codes = set(row[chop_codes_col].upper()).difference(set(valid_chop_codes))
        invalid_chop_codes = set(invalid_chop_codes)
        valid_codes_duplicated = list()
        for chop in valid_chop_codes:
            if chop in valid_chop_codes_deleted:
                valid_codes_duplicated.append(chop)

        discard_chops = invalid_chop_codes.union(set(valid_codes_duplicated))
        # set(row[chop_codes_col]).difference(set(chop_codes_deleted_col))
        # different_invalid_codes = set(row[chop_codes_col]).difference(set(invalid_chop_codes))
        #invalid_chop_codes_set = set(invalid_chop_codes)
        # discarded_chops = different_valid_codes.union(set(invalid_chop_codes))

        if len(discard_chops) > 0:
            logger.debug(f"row {row.name}: discarded duplicated and invalid CHOP entries after validation {discard_chops}")
        # if len(different_invalid_codes) > 0:
            # logger.debug(f"row {row.name}: discarded invalid CHOPs after validation {different_invalid_codes}")
            # logger.debug(f"row {row.name}: discarded valid CHOPs after validation {discarded_chops} discarded invalid CHOPs after validation {different_invalid_codes}")

        #if len(different_invalid_codes) > 0:
        #    logger.debug(f"row {row.name}: discarded invalid CHOPs after validation {different_invalid_codes}")

        row[output_chop_codes_col] = valid_chop_codes
        return row

    df = df.apply(_validate_chop_codes, axis=1)
    return df


def _remove_duplicates_case_insensitive(codes_list1: list[str], codes_list2: list[str]) -> (list[str], list[str]):
    """Compare 2 lists of CHOP codes, which are formatted as '<code>:<side>:<date>', and remove the codes which appear
    in both lists, regardless of their casing.

    @param codes_list1: The first list.
    @param codes_list2: The second list.
    @return: A tuple of (cleaned list 1, cleaned list 2), which do not contain the codes which appear in both.

    @note: If the intersection is empty, the original lists are returned, without being copied.
    """
    codes_list1_split = _split_chop_codes(codes_list1)
    codes_list2_split = _split_chop_codes(codes_list2)

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


def _split_chop_codes(codes_list: list[str]) -> list[list[str]]:
    """From a list of CHOP codes, which are formatted as '<code>:<side>:<date>', split them into their components.

    @param codes_list: The list of CHOP codes.
    @return: A list of the info for each code, split into strings.
    """
    return [code_with_colons.split(':') for code_with_colons in codes_list]


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
