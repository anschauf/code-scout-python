import unittest

from src.apps.extract_duration_of_stay_for_chops import compiled_concat_regex


class TestExtractDurationOfStayForChops(unittest.TestCase):
    def test_regex(self):
        description = 'Leberkomplexbehandlung, bis 6 Behandlungstage'
        matches = list(compiled_concat_regex.finditer(description))
        self.assertEqual(1, len(matches))



if __name__ == '__main__':
    unittest.main()
