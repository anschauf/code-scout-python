import os
from os.path import exists

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import PROJECT_ROOT_DIR
from src.schema import case_id_col
from src.service.bfs_cases_db_service import get_clinics, get_all_cases_socio_demographics_df, \
    get_all_revisions_df
from src.service.database import Database
from test.sandbox_hackathon.constants import RANDOM_SEED

def main(dir_output):
    if not exists(dir_output):
        os.makedirs(dir_output)

    with Database() as db:
        clinics = get_clinics(db.session)
        all_cases = get_all_cases_socio_demographics_df(db.session)
        all_revisions = get_all_revisions_df(db.session).groupby('aimedic_id', as_index=False)[['revision_id', 'drg', 'drg_cost_weight', 'effective_cost_weight', 'pccl', 'revision_date', 'adrg']].agg(lambda x: list(x))

        full_data = pd.merge(all_cases, all_revisions, how='outer', on='aimedic_id')
        full_data['y_label_is_revised_case'] = full_data['revision_id'].apply(lambda x: 1 if (len(x) > 1) else 0)

        # split data train test
        ind_train, ind_test = train_test_split(range(full_data.shape[0]), stratify=full_data['y_label_is_revised_case'].values, test_size=0.3, random_state=RANDOM_SEED)
        full_data_train = full_data.iloc[np.sort(ind_train)]
        full_data_test = full_data.iloc[np.sort(ind_test)]

        # write training and test data to file
        full_data_train[['aimedic_id', 'y_label_is_revised_case']].to_csv(os.path.join(dir_output, 'aimedic_id_label_data_train.csv'), index=False)
        full_data_test[['aimedic_id', 'y_label_is_revised_case']].to_csv(os.path.join(dir_output, 'aimedic_id_label_data_test.csv'), index=False)

        # prepare revised case file for model evaluation
        full_data_test_revised_cases = full_data_test[full_data_test['y_label_is_revised_case'] == 1].rename(columns={'aimedic_id': case_id_col})

        # leave the following empty, since for now we combine all those IDs and we use the aimedic_id now
        full_data_test_revised_cases['AdmNo'] = ['']*full_data_test_revised_cases.shape[0]
        full_data_test_revised_cases['FID'] = ['']*full_data_test_revised_cases.shape[0]
        full_data_test_revised_cases['PatID'] = ['']*full_data_test_revised_cases.shape[0]
        full_data_test_revised_cases['ICD_added'] = ['']*full_data_test_revised_cases.shape[0]
        full_data_test_revised_cases['ICD_dropped'] = ['']*full_data_test_revised_cases.shape[0]
        full_data_test_revised_cases['CHOP_added'] = ['']*full_data_test_revised_cases.shape[0]
        full_data_test_revised_cases['CHOP_dropped'] = ['']*full_data_test_revised_cases.shape[0]

        # extract revision data
        full_data_test_revised_cases['DRG_old'] = full_data_test_revised_cases['drg'].apply(lambda x: x[0])
        full_data_test_revised_cases['DRG_new'] = full_data_test_revised_cases['drg'].apply(lambda x: x[-1])
        full_data_test_revised_cases['CW_old'] = full_data_test_revised_cases['drg_cost_weight'].apply(lambda x: x[0])
        full_data_test_revised_cases['CW_new'] = full_data_test_revised_cases['drg_cost_weight'].apply(lambda x: x[-1])
        full_data_test_revised_cases['PCCL_old'] = full_data_test_revised_cases['pccl'].apply(lambda x: x[0])
        full_data_test_revised_cases['PCCL_new'] = full_data_test_revised_cases['pccl'].apply(lambda x: x[-1])

        full_data_test_revised_cases[[
            case_id_col,
            'AdmNo',
            'FID',
            'PatID',
            'ICD_added',
            'ICD_dropped',
            'CHOP_added',
            'CHOP_dropped',
            'DRG_old',
            'DRG_new',
            'CW_old',
            'CW_new',
            'PCCL_old',
            'PCCL_new'
        ]].to_csv(os.path.join(dir_output, 'aimedic_id_revised_cases.csv'), index=False)


if __name__ == "__main__":
    main(dir_output=os.path.join(PROJECT_ROOT_DIR, 'results', 'train_test_split'))