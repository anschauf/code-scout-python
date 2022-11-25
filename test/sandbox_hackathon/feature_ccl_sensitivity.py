import itertools
from os import makedirs
from os.path import join, exists

import awswrangler as wr
import numpy as np

from src import PROJECT_ROOT_DIR
from test.sandbox_hackathon.constants import FILENAME_TRAIN_SPLIT, FILENAME_TEST_SPLIT, RANDOM_SEED
from test.sandbox_hackathon.utils import load_data, train_lr_model, write_model_coefs_to_file, predict_proba, \
    write_evaluation_metrics_to_file, extract_case_ranking_performance_app, categorize_variable


def main():
    meta_data_train = wr.s3.read_csv(FILENAME_TRAIN_SPLIT)
    data_train_revised = load_data(meta_data_train, load_diagnoses=True, load_procedures=False, only_revised_cases=True)

    pccl_upgrade_idx = data_train_revised['pccl'].apply(lambda x: len(set(x)) > 1)
    data_train_revised_pccl = data_train_revised[pccl_upgrade_idx]

    def get_original_codes(row):
        idx_earliest_revision = np.argmin(row['revision_date'])
        id_earliest_revision = row['revision_id'][idx_earliest_revision]
        idx_earliest_revision_id = np.where(np.array(row['revision_id_diagnoses']) == id_earliest_revision)[0]

        all_icds = np.array(row['code_diagnoses'])[idx_earliest_revision_id]
        all_ccls = np.array(row['ccl_diagnoses'])[idx_earliest_revision_id]
        is_pd = np.array(row['is_primary_diagnoses'])[idx_earliest_revision_id]
        row['pd_icd'] = all_icds[is_pd][0]
        row['pd_ccl'] = all_ccls[is_pd][0]
        row['sd_icd'] = all_icds[~is_pd]
        row['sd_ccl'] = all_ccls[~is_pd]

        row['orig_drg'] = row['drg'][idx_earliest_revision]

        return row

    data_train_revised_pccl = data_train_revised_pccl.apply(get_original_codes, axis=1)

    n_codes = 1

    def calculate_delta_ccl(row):
        use_pd = row['orig_drg'].startswith('P')

        if row['aimedic_id'] == 737426:
            print()

        # Calculate the raw PCCL value
        original_pccl = calculate_raw_pccl(row['pd_ccl'], row['sd_ccl'], use_pd=use_pd)

        pccl_simulation = simulate_pccl_upgrade(row['pd_ccl'], row['sd_ccl'], use_pd=use_pd,
                                                original_pccl=original_pccl,
                                                n_codes=n_codes)

        if pccl_simulation is None:
            row[f'delta_pccl_upgrade_{n_codes}_codes'] = 0
        else:
            row[f'delta_pccl_upgrade_{n_codes}_codes'] = pccl_simulation

        return row

    data_train_revised_pccl = data_train_revised_pccl.apply(calculate_delta_ccl, axis=1)

    res = data_train_revised_pccl[['aimedic_id', 'drg', 'pd_icd', 'sd_icd', 'sd_ccl', 'pccl', f'delta_pccl_upgrade_{n_codes}_codes']]

    print('')


def simulate_pccl_upgrade(pd_ccl: int,
                          sd_ccls: list,
                          *,
                          original_pccl: float,
                          n_codes: int,
                          ccl_codes=(1, 2, 3, 4),
                          alpha: float = 0.4,
                          use_pd: bool = False) -> float:
    original_pccl_round = round(original_pccl)

    min_delta = None

    for n_codes_to_consider in range(1, n_codes + 1):
        for ccl_combination in itertools.combinations_with_replacement(ccl_codes, n_codes_to_consider):
            new_sd_ccls = list(sd_ccls)
            new_sd_ccls.extend(ccl_combination)
            upgraded_pccl = calculate_raw_pccl(pd_ccl, new_sd_ccls, alpha=alpha, use_pd=use_pd)

            upgraded_pccl_round = round(upgraded_pccl)
            if upgraded_pccl_round > original_pccl_round:
                delta_ccl_points = upgraded_pccl - original_pccl

                if min_delta is None:
                    min_delta = delta_ccl_points

                else:
                    if delta_ccl_points < min_delta:
                        min_delta = delta_ccl_points

    return min_delta


def calculate_raw_pccl(pd_ccl: int, sd_ccls: list,
                       *,
                       alpha: float = 0.4,
                       use_pd: bool = False,
                       ) -> float:
    if len(sd_ccls) == 0:
        return 0

    else:
        if use_pd:
            sorted_ccls = np.array([pd_ccl] + sorted(sd_ccls, reverse=True), dtype=int)
        else:
            sorted_ccls = np.array(sorted(sd_ccls, reverse=True), dtype=int)
        sorted_ccls = sorted_ccls[sorted_ccls > 0]

        indices = np.arange(sorted_ccls.shape[0])
        weighted_ccl = sorted_ccls * np.exp(-indices * alpha)
        scaled_ccl = np.log(np.sum(weighted_ccl) + 1) / (np.log(3 / alpha) / 4)
        return scaled_ccl






if __name__ == '__main__':
    main()
