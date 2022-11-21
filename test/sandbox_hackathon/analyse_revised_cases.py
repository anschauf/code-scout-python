from os import makedirs
from os.path import join, exists

import awswrangler as wr
import numpy as np
import pandas as pd

from src import PROJECT_ROOT_DIR
from test.sandbox_hackathon.constants import FILENAME_TRAIN_SPLIT, FILENAME_TEST_SPLIT
from test.sandbox_hackathon.utils import load_data


def main(dir_output):
    if not exists(dir_output):
        makedirs(dir_output)

    # load meta data containing aimedic id and label (whether the case was revised)
    meta_data_train = wr.s3.read_csv(FILENAME_TRAIN_SPLIT)
    meta_data_test = wr.s3.read_csv(FILENAME_TEST_SPLIT)

    # read in all data from DB and merge it with the labels from the meta data
    data = load_data(pd.concat([
        meta_data_train,
        meta_data_test
    ]).drop_duplicates("aimedic_id"), load_diagnoses=True, load_procedures=True, only_revised_cases=True)

    # create column added diagnoses for latest revision
    list_added_diagnoses = list()
    list_removed_diagnoses = list()
    list_added_procedures = list()
    list_removed_procedures = list()
    for row in data.itertuples():
        revision_id_original = row.revision_id[0]
        revision_id_revised = row.revision_id[-1]

        # diagnoses
        if revision_id_original in row.revision_id_diagnoses and revision_id_revised in row.revision_id_diagnoses:
            ind_original_icds = np.where(np.asarray(row.revision_id_diagnoses) == revision_id_original)[0]
            ind_revised_case_icds = np.where(np.asarray(row.revision_id_diagnoses) == revision_id_revised)[0]

            icds_original_case = np.asarray(row.code_diagnoses)[ind_original_icds]
            icds_revised_case = np.asarray(row.code_diagnoses)[ind_revised_case_icds]

            added_icds = set(icds_revised_case) - set(icds_original_case)
            list_added_diagnoses.append(list(added_icds))

            removed_icds = set(icds_original_case) - set(icds_revised_case)
            list_removed_diagnoses.append(list(removed_icds))

        else:
            list_added_diagnoses.append([])
            list_removed_diagnoses.append([])

        # chops
        if isinstance(row.revision_id_procedures, list) and revision_id_original in row.revision_id_procedures and revision_id_revised in row.revision_id_procedures:
            ind_original_chops = np.where(np.asarray(row.revision_id_procedures) == revision_id_original)[0]
            ind_revised_case_chops = np.where(np.asarray(row.revision_id_procedures) == revision_id_revised)[0]

            chops_original_case = np.asarray(row.code_procedures)[ind_original_chops]
            chops_revised_case = np.asarray(row.code_procedures)[ind_revised_case_chops]

            added_chops = set(chops_revised_case) - set(chops_original_case)
            list_added_procedures.append(list(added_chops))

            removed_chops = set(chops_original_case) - set(chops_revised_case)
            list_removed_procedures.append(list(removed_chops))

        else:
            list_added_procedures.append([])
            list_removed_procedures.append([])
    data['added_diagnoses'] = list_added_diagnoses
    data['removed_diagnoses'] = list_removed_diagnoses
    data['added_procedures'] = list_added_procedures
    data['removed_procedures'] = list_removed_procedures

    # count added and removed diagnoses
    unique_added_diagnoses, count_added_diagnoses = np.unique(np.concatenate(data['added_diagnoses'].values), return_counts=True)
    pd.DataFrame({'code': unique_added_diagnoses, 'count': count_added_diagnoses}).sort_values(by='count', ascending=False).to_csv(join(dir_output, 'count_added_diagnoses.csv'), index=False)
    unique_removed_diagnoses, count_removed_diagnoses = np.unique(np.concatenate(data['removed_diagnoses'].values), return_counts=True)
    pd.DataFrame({'code': unique_removed_diagnoses, 'count': count_removed_diagnoses}).sort_values(by='count', ascending=False).to_csv(join(dir_output, 'count_removed_diagnoses.csv'), index=False)

    # count added and removed chops
    unique_added_chops, count_added_chops = np.unique(np.concatenate(data['added_procedures'].values), return_counts=True)
    pd.DataFrame({'code': unique_added_chops, 'count': count_added_chops}).sort_values(by='count', ascending=False).to_csv(join(dir_output, 'count_added_procedures.csv'), index=False)
    unique_removed_chops, count_removed_chops = np.unique(np.concatenate(data['removed_procedures'].values), return_counts=True)
    pd.DataFrame({'code': unique_removed_chops, 'count': count_removed_chops}).sort_values(by='count', ascending=False).to_csv(join(dir_output, 'count_removed_procedures.csv'), index=False)


if __name__ == "__main__":
    main(dir_output=join(PROJECT_ROOT_DIR, 'results', 'analysis_revised_cases'))