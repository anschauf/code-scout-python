from unittest import TestCase

from src.service.bfs_cases_db_service import get_sociodemographics_for_hospital_year, get_earliest_revisions_for_aimedic_ids


class TestDbAccess(TestCase):
    def test_get_sociodemographics_for_hospital_year(self):
        df = get_sociodemographics_for_hospital_year('Hirslanden Aarau', 2018)
        self.assertTrue(df.shape[0] > 0)

    def test_get_earliest_revisions_for_aimedic_ids(self):
        df = get_earliest_revisions_for_aimedic_ids([120078, 119991])
        self.assertTrue(df.shape[0] > 0)
