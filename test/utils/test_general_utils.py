import unittest
from src.utils.general_utils import split_codes

class DataFrameUtilsTest(unittest.TestCase):

    def test_split_codes(self):
        ICD_code = str("I650|I652")
        CHOP_code = str("459812|461212|5413")
        split_code_ICD = split_codes(ICD_code)
        split_code_CHOP = split_codes(CHOP_code)
        self.assertListEqual(split_code_ICD, ['I650','I652'])
        self.assertListEqual(split_code_CHOP, ['459812','461212','5413'])
