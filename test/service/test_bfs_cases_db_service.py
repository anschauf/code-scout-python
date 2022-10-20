from unittest import TestCase

from src.service.bfs_cases_db_service import get_hospital_year_cases, get_sociodemographics_for_hospital_year


class TestDbAccess(TestCase):
    def test_get_sociodemographics_for_hospital_year(self):
        df = get_sociodemographics_for_hospital_year('Hirslanden Aarau', 2018)
        self.assertTrue(df.shape[0] > 0)


