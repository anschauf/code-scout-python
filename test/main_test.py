import unittest

from src.apps.code_ranking_tier import calculate_performance


class MyTestCase(unittest.TestCase):
    def test_calculate_performance(self):
        calculate_performance()


if __name__ == '__main__':
    unittest.main()
