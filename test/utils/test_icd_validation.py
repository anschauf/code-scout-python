# from unittest import TestCase
#
# from src.utils.icd_validation import validate_icd_codes_list
#
#
# class TestIcdValidation(TestCase):
#     def test_validate_icd_codes_list__invalid_codes(self):
#         icds = ['abc', '12345', 'i109a', 'i10901']
#         valid_icds = validate_icd_codes_list(icds)
#         self.assertEqual(len(valid_icds), 0)
#
#     def test_validate_icd_codes_list__valid_codes(self):
#         icds = ['i1090', 'f03']
#         valid_icds = validate_icd_codes_list(icds)
#
#         self.assertListEqual(valid_icds, ['I1090', 'F03'])
