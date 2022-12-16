import numpy as np

from test.sandbox_model_case_predictions.data_handler import load_data
from test.sandbox_model_case_predictions.utils import get_list_of_all_predictors

all_data = load_data()

list_X, list_X_labels = get_list_of_all_predictors(all_data)

X = np.hstack(list_X)
X_labels = np.concatenate(list_X_labels)
y = np.zeros((X.shape[0], ))


print('')
