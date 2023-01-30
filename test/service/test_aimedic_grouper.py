import unittest
import pandas as pd

from src.service.aimedic_grouper import AIMEDIC_GROUPER
from src.service.bfs_cases_db_service import get_sociodemographics_for_hospital_year, \
    get_original_revision_id_for_sociodemographic_ids, get_revised_case_with_codes_before_revision, \
    get_revised_case_with_codes_after_revision
from src.utils.global_configs import GROUPER_FORMAT_COL
from src.utils.group import format_for_grouper

from src.service.database import Database

class AimedicGrouperTest(unittest.TestCase):
    def test_grouper_jar_is_callable(self):
        revised_cases_df = pd.DataFrame([['100',
                      '21320891027', 'R21', ['J91', 'I318', 'E788', 'I1090', 'J9580'], '371211:L:20170216', ['340999:L:20170216', '009910::20170216', '3491:L:20170219'],
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



        sd_orginal = ['I1090' ,'I481' ,'Z921' ,'G406' ,'I8720' ,'Z867' ,'Z950']
        sd_add = ''
    def test_suggested_codes_grouper(self):
        with Database() as db:
            sociodemo_hospital_year = get_sociodemographics_for_hospital_year(
                hospital_name='Kantonsspital Winterthur', year=2020, session=db.session)
            sociodemographic_id = sociodemo_hospital_year['sociodemographic_id'].tolist()
            sociodemographic_id_revision_id_df = get_original_revision_id_for_sociodemographic_ids(
                sociodemographic_id, db.session)

            # case need to be tested
            revised_cases_before_revision = get_revised_case_with_codes_before_revision(db.session)
            revised_cases_after_revision = get_revised_case_with_codes_after_revision(db.session)
            revised_cases_sociodemographic_id = revised_cases_before_revision['sociodemographic_id'].tolist()
            revised_cases_sociodemographic_id_ksw = [sociodemo_id for sociodemo_id in
                                                     revised_cases_sociodemographic_id
                                                     if sociodemo_id in sociodemographic_id]

            revised_cases_before_revision_ksw_2020 = revised_cases_before_revision[
                revised_cases_before_revision['sociodemographic_id'].isin(revised_cases_sociodemographic_id_ksw)]
            revised_cases_after_revision_ksw_2020 = revised_cases_after_revision[
                revised_cases_after_revision['sociodemographic_id'].isin(revised_cases_sociodemographic_id_ksw)]

            print('')
            # Test 5 revised cases
            sociodemo_id_revised_5 = revised_cases_before_revision_ksw_2020['sociodemographic_id'].tolist()[:5]
            revised_cases_before_revision_5 = revised_cases_before_revision_ksw_2020[
                revised_cases_before_revision_ksw_2020['sociodemographic_id'].isin(sociodemo_id_revised_5)]
            revision_id_5 = revised_cases_before_revision_5['revision_id'].tolist()
            sociodemo_info_5 = sociodemo_hospital_year[
                sociodemo_hospital_year['sociodemographic_id'].isin(sociodemo_id_revised_5)]

            revised_cases_before_revision_5.set_index('sociodemographic_id', inplace=True)
            sociodemo_info_5.set_index('sociodemographic_id', inplace=True)

            all_info_case_5 = pd.concat(
                [sociodemo_info_5, revised_cases_before_revision_5], axis=1)
            all_info_case_5.reset_index(inplace=True)
            print(all_info_case_5.columns)
            colums_for_grouper = all_info_case_5[['sociodemographic_id', 'case_id', 'primary_diagnosis', 'secondary_diagnoses', 'primary_procedure',
                     'secondary_procedures', 'gender', 'age_years', 'age_days', 'gestation_age', 'duration_of_stay',
                     'ventilation_hours', 'grouper_admission_type', 'admission_date', 'admission_weight', 'grouper_discharge_type', 'discharge_date', 'medications']]
            # colums_for_grouper = all_info_case_5[['sociodemographic_id', 'case_id']]
            # Prepare the format for grouper
            revision_id = 816275
            sociodemographic_id = 816247
            socio_demo_case = sociodemo_hospital_year[sociodemo_hospital_year['']]

            print('')
            # df = pd.DataFrame()

if __name__ == '__main__':
    unittest.main()
