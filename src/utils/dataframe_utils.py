import pandas as pd
from beartype import beartype


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
    def _remove_duplicated_chops(s):
        s[cleaned_added_chops_col], s[cleaned_removed_chops_col] = _remove_duplicates_case_insensitive(s[added_chops_col], s[removed_chops_col])
        return s

    df = df.apply(_remove_duplicated_chops, axis=1)
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
