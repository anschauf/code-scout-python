import unittest

import pandas as pd

from src.service.aimedic_grouper import group_batch_group_cases
from src.revised_case_normalization.notebook_functions.group import format_for_grouper
from src.revised_case_normalization.notebook_functions.global_configs import GROUPER_FORMAT_COL


class AimedicGrouperTest(unittest.TestCase):
    def test_grouper_jar_is_callable(self):
        revised_cases_df = pd.DataFrame([['HLAA|BUR#71275707|FileDate:20180227#1|Pat:07C27DA4B851FD42|Case:0041282182|in2017-out2017|67Y|W|PD:I313|SD:E788,I1090,I318,J91|SRG:009910,340999,3491,371211',
                      '0041282182', 'I313', ['J91', 'I318', 'E788', 'I1090', 'J9580'], '371211:L:20170216', ['340999:L:20170216', '009910::20170216', '3491:L:20170219'],
                      'W', 67, 0, 0, 7, 0, '01', '2017-02-16', 0, 00, '2017-02-23']],
                      columns=['aimedic_id', 'case_id', 'primary_diagnosis', 'secondary_diagnoses', 'primary_procedure',
                     'secondary_procedures', 'gender', 'age_years', 'age_days', 'gestation_age', 'duration_of_stay',
                     'ventilation_hours', 'grouper_admission_type', 'admission_date', 'admission_weight', 'grouper_discharge_type', 'discharge_date'])
        formatted = format_for_grouper(revised_cases_df)
        grouper_format = formatted.loc[0][GROUPER_FORMAT_COL]
        self.assertEqual(grouper_format, 'HLAA|BUR#71275707|FileDate:20180227#1|Pat:07C27DA4B851FD42|Case:0041282182|in2017-out2017|67Y|W|PD:I313|SD:E788,I1090,I318,J91|SRG:009910,340999,3491,371211;41282182;67;0;;W;20170216;01;20170223;0;7;0;I313|J91|I318|E788|I1090|J9580;371211:L:20170216|340999:L:20170216|009910::20170216|3491:L:20170219;')
        df1, df2, df3 = group_batch_group_cases([grouper_format])

        print("")


if __name__ == '__main__':
    unittest.main()
