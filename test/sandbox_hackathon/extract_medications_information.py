# drg, count, diagnoses, chops, medication (counts for all of them)
from os import makedirs
from os.path import join, exists

import awswrangler as wr
import numpy as np
import pandas as pd

from src import PROJECT_ROOT_DIR
from test.sandbox_hackathon.constants import FILENAME_TRAIN_SPLIT, FILENAME_TEST_SPLIT
from test.sandbox_hackathon.utils import load_data, get_revision_id_of_original_case


def main(dir_output):
    if not exists(dir_output):
        makedirs(dir_output)

    meta_data_train = wr.s3.read_csv(FILENAME_TRAIN_SPLIT)
    meta_data_test = wr.s3.read_csv(FILENAME_TEST_SPLIT)

    data = load_data(pd.concat(
        [meta_data_test, meta_data_train]
    ), load_diagnoses=True, load_procedures=True, only_revised_cases=False)

    data_medications = data[data['medications'] != '']
    drg_unique, drg_counts = np.unique(np.concatenate(data_medications['drg'].values), return_counts=True)
    medications = pd.DataFrame({'drg': drg_unique, 'counts': drg_counts}).sort_values(by='counts', ascending=False)

    def concat_counts(x):
        return '%s (%i)' %(x[0], x[1])

    list_diagnoses = list()
    list_chops = list()
    list_medications = list()
    for row in medications.itertuples():
        drg = row.drg
        data_medications_drg = data_medications[data_medications['drg'].apply(lambda x: drg in x)]

        # add medications
        medications_drg = np.concatenate([x.split("|") for x in data_medications_drg['medications'].values])
        medications_drg_unique, medications_drg_count = np.unique(medications_drg, return_counts=True)
        medications_drg_counts_df = pd.DataFrame({0: medications_drg_unique, 1: medications_drg_count}).sort_values(by=1, ascending=False)
        medications_drg_counts_df['output'] = medications_drg_counts_df.apply(concat_counts, axis=1)
        list_medications.append(' | '.join(medications_drg_counts_df['output'].values))

        list_all_chops_drg = list()
        list_all_diags_drg = list()
        for row_case in data_medications_drg.itertuples():
            ind_original, revision_id = get_revision_id_of_original_case(row_case)

            # add procedures
            ind_chops = np.where(np.asarray(row_case.revision_id_procedures) == revision_id)[0]
            if len(ind_chops) > 0:
                list_all_chops_drg.append(np.unique(np.asarray(row_case.code_procedures)[ind_chops]))

            # add diagnoses
            ind_diags = np.where(np.asarray(row_case.revision_id_diagnoses) == revision_id)[0]
            if len(ind_diags) > 0:
                list_all_diags_drg.append(np.unique(np.asarray(row_case.code_diagnoses)[ind_diags]))

        # get unique procedure counts
        if len(list_all_chops_drg) > 0:
            chops_drg_unique, chops_drg_counts = np.unique(np.concatenate(list_all_chops_drg), return_counts=True)
            chops_drg_counts_df = pd.DataFrame({0: chops_drg_unique, 1: chops_drg_counts}).sort_values(by=1, ascending=False)
            chops_drg_counts_df['output'] = chops_drg_counts_df.apply(concat_counts, axis=1)
            list_chops.append(' | '.join(chops_drg_counts_df['output'].values))
        else:
            list_chops.append([''])

        # get unique diagnoses counts
        if len(list_all_diags_drg) > 0:
            diags_drg_unique, diags_drg_counts = np.unique(np.concatenate(list_all_diags_drg), return_counts=True)
            diags_drg_counts_df = pd.DataFrame({0: diags_drg_unique, 1: diags_drg_counts}).sort_values(by=1, ascending=False)
            diags_drg_counts_df['output'] = diags_drg_counts_df.apply(concat_counts, axis=1)
            list_diagnoses.append(' | '.join(diags_drg_counts_df['output'].values))
        else:
            list_diagnoses.append([''])

    medications['medications'] = list_medications
    medications['diagnoses'] = list_diagnoses
    medications['procedures'] = list_chops
    medications.to_csv(join(dir_output, 'drg_counts_with_medications.csv'), index=False)

if __name__ == "__main__":
    main(dir_output=join(PROJECT_ROOT_DIR, 'results', 'medication_counts'))