import unittest

import pandas as pd

# noinspection PyProtectedMember
from src.utils.dataframe_utils import _remove_duplicates_case_insensitive, remove_duplicated_chops, validate_icd_codes


class DataFrameUtilsTest(unittest.TestCase):
    def test_remove_duplicates_case_insensitive__no_letters_no_duplicates(self):
        added_chops = ['101010::20001010', '121212::20001010']
        removed_chops = ['202020::20001010', '323232::20001010']
        cleaned_added_chops, cleaned_removed_chops = _remove_duplicates_case_insensitive(added_chops, removed_chops)

        self.assertListEqual(added_chops, cleaned_added_chops)
        self.assertListEqual(removed_chops, cleaned_removed_chops)

    def test_remove_duplicates_case_insensitive__with_letters_no_duplicates(self):
        added_chops = ['1a1a1a::20001010', '1B1B1B::20001010']
        removed_chops = ['2a2a2a::20001010', '3B3B3B::20001010']
        cleaned_added_chops, cleaned_removed_chops = _remove_duplicates_case_insensitive(added_chops, removed_chops)

        self.assertListEqual(added_chops, cleaned_added_chops)
        self.assertListEqual(removed_chops, cleaned_removed_chops)

    def test_remove_duplicates_case_insensitive__no_letters_with_duplicates(self):
        added_chops = ['101010::20001010', '121212::20001010']
        removed_chops = ['202020::20001010', '121212::20001010']
        cleaned_added_chops, cleaned_removed_chops = _remove_duplicates_case_insensitive(added_chops, removed_chops)

        self.assertListEqual(cleaned_added_chops, [added_chops[0]])
        self.assertListEqual(cleaned_removed_chops, [removed_chops[0]])

    def test_remove_duplicates_case_insensitive__with_letters_with_duplicates(self):
        added_chops = ['1A1A1A::20001010', '1b1b1b::20001010']
        removed_chops = ['2A2A2A::20001010', '1b1b1b::20001010']
        cleaned_added_chops, cleaned_removed_chops = _remove_duplicates_case_insensitive(added_chops, removed_chops)

        self.assertListEqual(cleaned_added_chops, [added_chops[0]])
        self.assertListEqual(cleaned_removed_chops, [removed_chops[0]])

    def test_remove_duplicated_chops(self):
        added_chops = ['1A1A1A::20001010', '1b1b1b::20001010']
        removed_chops = ['2A2A2A::20001010', '1b1b1b::20001010']
        df = pd.DataFrame([[added_chops], [removed_chops]]).T
        df.columns = ['added_chops', 'removed_chops']

        df = remove_duplicated_chops(df)
        first_row = df.loc[0]
        self.assertListEqual(first_row['cleaned_added_chops'], [added_chops[0]])
        self.assertListEqual(first_row['cleaned_removed_chops'], [removed_chops[0]])

    def test_validate_icd_codes(self):
        invalid_icds = ['abc', '12345', 'i109a', 'i10901']
        valid_icds = ['i1090', 'f03']
        added_icds = invalid_icds + valid_icds
        df = pd.DataFrame([[added_icds]]).T
        df.columns = ['added_icds']

        df = validate_icd_codes(df)
        first_row = df.loc[0]
        self.assertListEqual(first_row['added_icds'], ['I1090', 'F03'])


if __name__ == '__main__':
    unittest.main()
