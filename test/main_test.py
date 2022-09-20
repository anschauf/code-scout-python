import unittest

from src.apps.code_ranking_tiers import calculate_code_ranking_performance


class MyTestCase(unittest.TestCase):
    def test_calculate_performance(self):
        calculate_code_ranking_performance()


if __name__ == '__main__':
    unittest.main()
