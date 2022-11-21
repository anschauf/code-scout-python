from unittest import TestCase

import pandas as pd

from src.revised_case_normalization.notebook_functions.global_configs import AIMEDIC_ID_COL
# from src.revised_case_normalization.notebook_functions.revise import apply_revisions
from src.service.bfs_cases_db_service import get_sociodemographics_for_hospital_year
# , get_earliest_revisions_for_aimedic_ids, insert_revised_cases_into_diagnoses, \
#     get_diagnoses_codes, get_procedures_codes, get_codes, insert_revised_cases_into_revisions
# from src.service.bfs_cases_db_service import insert_revised_cases_into_procedures, get_bfs_cases_by_ids
# from src.service.database import Database
#
# # TODO: This instance should be the test DB
# database = Database()
#
#
class TestDbAccess(TestCase):
    def test_get_bfs_cases_by_ids(self):
        with database as db:
            data = get_bfs_cases_by_ids(['40985494'], db.session)
        self.assertIsInstance(data, pd.DataFrame)

    def test_get_sociodemographics_for_hospital_year(self):
        with database as db:
            df = get_sociodemographics_for_hospital_year('Hirslanden Aarau', 2018, db.session)
        num_columns = len(df.columns)
        self.assertequal(num_columns, 34)
        self.assertTrue(df.shape[0] > 0)

#     def test_get_earliest_revisions_for_aimedic_ids(self):
#         with database as db:
#             df = get_earliest_revisions_for_aimedic_ids([115875], db.session)
#         self.assertTrue(df.shape[0] > 0)
#         self.assertListEqual(list(df.columns), [AIMEDIC_ID_COL, 'revision_id'])
#
#     def test_get_diagnoses_codes(self):
#         with database as db:
#             df_revision_ids = get_earliest_revisions_for_aimedic_ids([120078, 119991], db.session)
#             df = get_diagnoses_codes(df_revision_ids, db.session)
#         self.assertTrue(df.shape[0] > 0)
#
#     def test_get_procedures_codes(self):
#         with database as db:
#             df_revision_ids = get_earliest_revisions_for_aimedic_ids([1, 2], db.session)
#             df = get_procedures_codes(df_revision_ids, db.session)
#         self.assertTrue(df.shape[0] > 0)
#
#     def test_get_codes(self):
#         with database as db:
#             df_revision_ids = get_earliest_revisions_for_aimedic_ids([120078, 119991], db.session)
#             df = get_codes(df_revision_ids, db.session)
#         self.assertTrue(df.shape[0] > 0)
#
#     def test_apply_revisions(self):
#         # Test whether we can add and remove codes, according to some revisions.
#         # This test will check:
#         #   1. Remove a CHOP code which appears twice in the case
#         #   2. Handle the grouper format correctly
#         #   3. Add ICDs and CHOPs
#
#         cases_df = pd.DataFrame([[1, 1, 'I7024', ['I7020', 'Z9588', 'J4483'], '395014:L:20000101', ['395010:R:20000202', '395011:R:20000202', '395011:R:20000203']]],
#                                 columns=['aimedic_id', 'revision_id', 'old_pd', 'secondary_diagnoses', 'primary_procedure', 'secondary_procedures'])
#
#         revisions_df = pd.DataFrame([[1, 'I7024', ['J4481'], ['J4483'], ['395024'], ['395011']]],
#                                     columns=['aimedic_id', 'primary_diagnosis', 'added_icds', 'removed_icds', 'added_chops', 'removed_chops'])
#
#         revised_cases = apply_revisions(cases_df, revisions_df)
#
#         # Extract the first row and test the output
#         row = revised_cases.loc[0]
#
#         self.assertListEqual(row['secondary_diagnoses'], ['I7020', 'Z9588', 'J4481'])
#         self.assertEqual(row['primary_procedure'], '395014:L:20000101')
#         self.assertListEqual(row['secondary_procedures'], ['395010:R:20000202', '395024::'])
#
#     def test_insert_revised_cases_into_revisions(self):
#         revision_df = pd.DataFrame([[1, 'G07Z', 0.984, 0.65, 0, '2024-12-31'],
#                                     [2, 'F59B', 2.549,	1.495, 4, '2024-12-31']],
#                                    columns=['aimedic_id', 'drg', 'drg_cost_weight', 'effective_cost_weight', 'pccl', 'revision_date'])
#         with database as db:
#             aimedic_id_revision_id = insert_revised_cases_into_revisions(revision_df, db.session)
#
#         self.assertEqual(len(aimedic_id_revision_id), revision_df.shape[0])
#         self.assertIsInstance(aimedic_id_revision_id, dict)
#
#     def test_insert_revised_cases_into_diagnoses(self):
#         diagnoses_df = pd.DataFrame([
#             [1, 'Z432', 0, False, False],
#             [1, 'I440', 0, False, False]],
#             columns=['aimedic_id', 'code', 'ccl', 'is_primary', 'is_grouper_relevant'])
#         aimedic_id_revision_id = {1: 1, 2: 2}
#         with database as db:
#             insert_revised_cases_into_diagnoses(diagnoses_df, aimedic_id_revision_id, db.session)
#         self.assertTrue(True)
#
#     def test_insert_revised_cases_into_procedures(self):
#         procedures_df = pd.DataFrame([[1, 893909, '', '2024-12-31', False, True],
#                                      [1, 887910, 'B', '2024-12-31', False, False]],
#                                      columns=['aimedic_id', 'code', 'side', 'date', 'is_grouper_relevant', 'is_primary'])
#         aimedic_id_revision_id = {1: 1, 2: 2}
#         with database as db:
#             insert_revised_cases_into_procedures(procedures_df, aimedic_id_revision_id, db.session)
#         self.assertTrue(True)
