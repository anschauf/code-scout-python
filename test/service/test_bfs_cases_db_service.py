from unittest import TestCase

import pandas as pd

from src.service.bfs_cases_db_service import get_sociodemographics_for_hospital_year, get_earliest_revisions_for_aimedic_ids, \
    get_diagnoses_codes, get_procedures_codes, get_codes, apply_revisions
from src.revised_case_normalization.py.global_configs import AIMEDIC_ID_COL


class TestDbAccess(TestCase):
    def test_get_sociodemographics_for_hospital_year(self):
        df = get_sociodemographics_for_hospital_year('Hirslanden Aarau', 2018)
        self.assertTrue(df.shape[0] > 0)

    def test_get_earliest_revisions_for_aimedic_ids(self):
        df = get_earliest_revisions_for_aimedic_ids([120078, 119991])
        self.assertTrue(df.shape[0] > 0)
        self.assertListEqual(list(df.columns), [AIMEDIC_ID_COL, 'revision_id'])

    def test_get_diagnoses_codes(self):
        df_revision_ids = get_earliest_revisions_for_aimedic_ids([120078, 119991])
        df = get_diagnoses_codes(df_revision_ids)
        self.assertTrue(df.shape[0] > 0)

    def test_get_procedures_codes(self):
        df_revision_ids = get_earliest_revisions_for_aimedic_ids([120078, 119991])
        df = get_procedures_codes(df_revision_ids)
        self.assertTrue(df.shape[0] > 0)

    def test_get_codes(self):
        df_revision_ids = get_earliest_revisions_for_aimedic_ids([120078, 119991])
        df = get_codes(df_revision_ids)
        self.assertTrue(df.shape[0] > 0)

    def test_apply_revisions(self):
        cases_df = pd.DataFrame([[1, 1, 'I7024', ['I7020', 'Z9588', 'J4483'], '395014', ['395011']]],
                                columns=['aimedic_id', 'revision_id', 'old_pd', 'secondary_diagnoses', 'primary_procedure', 'secondary_procedures'])

        revisions_df = pd.DataFrame([[1, 'I7024', ['J4481'], ['J4483'], [], []]],
                                    columns=['aimedic_id', 'primary_diagnosis', 'added_icds', 'removed_icds', 'added_chops', 'removed_chops'])

        revised_cases = apply_revisions(cases_df, revisions_df)
        self.assertTrue(revised_cases.shape[0] > 0)
