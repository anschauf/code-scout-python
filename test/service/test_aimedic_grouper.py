import unittest

import pandas as pd

from src.service.aimedic_grouper import AIMEDIC_GROUPER
from src.utils.global_configs import GROUPER_FORMAT_COL
from src.utils.group import format_for_grouper


class AimedicGrouperTest(unittest.TestCase):
    def test_grouper_jar_is_callable(self):
        revised_cases_df = pd.DataFrame([['100',
                      '0041282182', 'I313', ['J91', 'I318', 'E788', 'I1090', 'J9580'], '371211:L:20170216', ['340999:L:20170216', '009910::20170216', '3491:L:20170219'],
                      'W', 67, 0, 0, 7, 0, '01', '2017-02-16', 0, 00, '2017-02-23', '']],
                      columns=['sociodemographic_id', 'case_id', 'primary_diagnosis', 'secondary_diagnoses', 'primary_procedure',
                     'secondary_procedures', 'gender', 'age_years', 'age_days', 'gestation_age', 'duration_of_stay',
                     'ventilation_hours', 'grouper_admission_type', 'admission_date', 'admission_weight', 'grouper_discharge_type', 'discharge_date', 'medications'])
        formatted = format_for_grouper(revised_cases_df)
        grouper_format = formatted.loc[0][GROUPER_FORMAT_COL]
        self.assertEqual(grouper_format, '100;67;0;;W;20170216;01;20170223;0;7;0;I313|J91|I318|E788|I1090|J9580;371211:L:20170216|340999:L:20170216|009910::20170216|3491:L:20170219;')

        # Create an additional case, with a different ID
        grouper_format2 = '2' + grouper_format[1:]

        df = AIMEDIC_GROUPER.run_batch_grouper(cases=[grouper_format, grouper_format2])
        self.assertEqual(df.shape[0], 2)


if __name__ == '__main__':
    unittest.main()
