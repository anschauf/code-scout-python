import numpy as np
from loguru import logger

from test.sandbox_hackathon.utils import get_revision_id_of_original_case, categorize_variable


def extract_number_of(data):
    # create number of code fields
    X_number_of_sdx = np.zeros((len(data), 1))
    X_number_of_chops = np.zeros((len(data), 1))
    X_number_of_used_diags = np.zeros((len(data), 1))
    X_number_of_used_chops = np.zeros((len(data), 1))
    X_number_of_ccl_triggering_diags = np.zeros((len(data), 1))
    original_pccl = np.asarray(['']*len(data)).reshape((-1,1))
    original_effective_cost_weight = np.zeros((len(data), 1))
    for i, row in enumerate(data.itertuples()):
        ind_original_case, revision_id = get_revision_id_of_original_case(row)
        original_pccl[i] = np.asarray(row.pccl)[ind_original_case]
        original_effective_cost_weight[i] = np.asarray(row.effective_cost_weight)[ind_original_case]
        ind_diags = np.where(np.asarray(row.revision_id_diagnoses) == revision_id)[0]
        ind_chops = np.where(np.asarray(row.revision_id_procedures) == revision_id)[0]

        if len(ind_diags) > 0:
            # number of side diagnoses
            X_number_of_sdx[i] = len(ind_diags) - 1
            # number of used diags
            X_number_of_used_diags[i] = np.sum(np.asarray(row.is_grouper_relevant_diagnoses)[ind_diags])
            # number of ccl triggering diags
            X_number_of_ccl_triggering_diags[i] = np.sum(np.asarray(row.ccl_diagnoses)[ind_diags] > 0)

        if len(ind_chops) > 0:
            # number of chops
            X_number_of_chops[i] = len(ind_chops)
            # number of used chops
            X_number_of_used_chops[i] = np.sum(np.asarray(row.is_grouper_relevant_procedures)[ind_chops])

    predictor_labels = ['number_of_sdx', 'number_of_chops', 'number_of_used_diags', 'number_of_used_chops', 'number_of_ccl_triggering_diags']

    return X_number_of_sdx, X_number_of_chops, X_number_of_used_diags, X_number_of_used_chops, X_number_of_ccl_triggering_diags, predictor_labels, original_pccl, original_effective_cost_weight, \

def get_categorized_predictors(data_train, data_test, column):
    logger.info(f'Preparing categorical variable for column {column}')
    X_train, label, encoder = categorize_variable(data_train, column)
    X_test, _, _ = categorize_variable(data_test, column, encoder=encoder)
    return X_train, X_test, label


