import unittest

import pandas as pd

from src.utils.global_configs import GROUPER_FORMAT_COL
from src.utils.group import format_for_grouper, group_revised_cases_for_db


class GroupTest(unittest.TestCase):
    def test_format_for_grouper(self):
        revised_cases_df = pd.DataFrame([[
            2, 1, 'I7024', ['I7020', 'Z9588', 'I1090', 'N184', 'N390'], '395014:R:20180111', ['395011:R:20180111', '397510:R:20180111', '004C12::20180111', '005599:R:20180111', '0043:R:20180111'],
            'W', 77, 0, 0, 2, 0, '01', '2018-01-10', 0, '00', '2018-01-12', ''
        ]], columns=['sociodemographic_id', 'case_id', 'primary_diagnosis', 'secondary_diagnoses', 'primary_procedure', 'secondary_procedures', 'gender', 'age_years', 'age_days', 'gestation_age', 'duration_of_stay', 'ventilation_hours', 'grouper_admission_type', 'admission_date', 'admission_weight', 'grouper_discharge_type', 'discharge_date', 'medications'])

        formatted = format_for_grouper(revised_cases_df)
        grouper_format = formatted.loc[0][GROUPER_FORMAT_COL]
        self.assertEqual(grouper_format, '2;77;0;;W;20180110;01;20180112;00;2;0;I7024|I7020|Z9588|I1090|N184|N390;395014:R:20180111|395011:R:20180111|397510:R:20180111|004C12::20180111|005599:R:20180111|0043:R:20180111;')

    def test_group(self):
        revised_cases_df = pd.DataFrame([[
            2, 1, 'I7024', ['I7020', 'Z9588', 'I1090', 'N184', 'N390'], '395014:R:20180111', ['395011:R:20180111', '397510:R:20180111', '004C12::20180111', '005599:R:20180111', '0043:R:20180111'],
            'W', 77, 0, 0, 2, 0, '01', '2018-01-10', 0, '00', '2018-01-12', ''
        ]], columns=['sociodemographic_id', 'case_id', 'primary_diagnosis', 'secondary_diagnoses', 'primary_procedure', 'secondary_procedures', 'gender', 'age_years', 'age_days', 'gestation_age', 'duration_of_stay', 'ventilation_hours', 'grouper_admission_type', 'admission_date', 'admission_weight', 'grouper_discharge_type', 'discharge_date', 'medications'])

        revision_df, diagnoses_df, procedures_df = group_revised_cases_for_db(revised_cases_df)

        self.assertEqual(revision_df.shape[0], 1)
        self.assertEqual(diagnoses_df.shape[0], 6)
        self.assertEqual(procedures_df.shape[0], 6)


if __name__ == '__main__':
    unittest.main()
