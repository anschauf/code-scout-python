from unittest import TestCase

from src.utils.chop_validation import validate_chop_codes_list


class TestChopValidation(TestCase):
    def test_validate_chop_codes_list__invalid_codes(self):
        chops = ['aa', 'a1', '8311.807799']
        valid_chops = validate_chop_codes_list(chops)
        self.assertEqual(len(valid_chops), 0)

    def test_validate_chop_codes_list__valid_codes(self):
        chops = ['1a', '7a4412', '807799', '8311']
        valid_chops = validate_chop_codes_list(chops)
        self.assertListEqual(valid_chops, ['1A', '7A4412', '807799', '8311'])
