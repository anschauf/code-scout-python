from unittest import TestCase

from sandbox_model_case_predictions.utils import list_all_subsets


class Test(TestCase):
    def test_list_all_subsets(self):
        n_features = 3
        subsets = list(list_all_subsets(range(n_features)))
        reversed_subsets = list(list_all_subsets(range(n_features), reverse=True))

        print('')




