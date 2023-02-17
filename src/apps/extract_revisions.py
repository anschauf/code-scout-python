import itertools
import itertools
import os.path

import pandas as pd
from loguru import logger

from src import ROOT_DIR
from src.service.bfs_cases_db_service import get_hospitals, get_revised_case_with_codes_after_revision, \
    get_revised_case_with_codes_before_revision, get_sociodemographics_by_sociodemographics_ids
from src.service.database import Database


def extract_revisions():
    with Database() as db:
        logger.info('Reading the revised cases before revision ...')
        original_cases = get_revised_case_with_codes_before_revision(db.session).reset_index(drop=False)

        sorted_columns = sorted(list(original_cases.columns))
        sorted_columns.remove('sociodemographic_id')
        sorted_columns.insert(0, 'sociodemographic_id')

        logger.info('Reading the sociodemographic data of these cases ...')
        original_cases_ids = original_cases['sociodemographic_id'].values.tolist()
        sociodemographics = get_sociodemographics_by_sociodemographics_ids(original_cases_ids, db.session)[
            ['sociodemographic_id', 'case_id', 'hospital_id']]
        hospitals = get_hospitals(db.session)[['hospital_id', 'hospital_abbreviation']].rename(
            columns={'hospital_abbreviation': 'hospital'})
        sociodemographics = pd.merge(sociodemographics, hospitals, on='hospital_id').drop(columns=['hospital_id'])

        logger.info('Reading the revised cases after revision ...')
        revised_cases = get_revised_case_with_codes_after_revision(db.session).reset_index(drop=False)

    logger.info('Joining all the info ...')
    df = pd.merge(original_cases, revised_cases, on='sociodemographic_id', suffixes=('_original', '_revised'))
    df = pd.merge(df, sociodemographics, on='sociodemographic_id')
    df = df.sort_values(by='sociodemographic_id', ascending=True).reset_index(drop=True)

    sorted_column_pairs = ['sociodemographic_id', 'case_id', 'hospital']
    for col_name in sorted_columns[1:]:
        sorted_column_pairs.append(f'{col_name}_original')
        sorted_column_pairs.append(f'{col_name}_revised')
    df = df[sorted_column_pairs]

    def _extract_revision(row):
        original_diagnoses = row['all_diagnoses_original']
        revised_diagnoses = row['all_diagnoses_revised']

        diff_diagnoses = tuple(set(revised_diagnoses).difference(set(original_diagnoses)))
        is_grouper_relevant_diagnoses_revised = row['is_grouper_relevant_diagnoses_revised']

        add_diagnoses_true = [diag for diag in diff_diagnoses if diag not in original_diagnoses]

        if len(diff_diagnoses) != len(add_diagnoses_true):
            case_id = row['case_id']
            logger.info(f'Case {case_id} has deleted diagnoses')
        idx_add_diagnoses = [revised_diagnoses.index(added_diag) for added_diag in add_diagnoses_true]

        row['added_diagnoses'] = add_diagnoses_true
        row['added_diagnoses_grouper_relevant'] = [is_grouper_relevant_diagnoses_revised[idx] for idx in idx_add_diagnoses]

        original_procedures = row['all_procedures_original']
        revised_procedures = row['all_procedures_revised']

        original_procedures = [p.split(':')[0] for p in original_procedures]
        original_procedures = [p for p in original_procedures if p != '' and p != 'nan']

        revised_procedures = [p.split(':')[0] for p in revised_procedures]
        revised_procedures = [p for p in revised_procedures if p != '' and p != 'nan']

        diff_procedures = tuple(set(revised_procedures).difference(set(original_procedures)))
        is_grouper_relevant_procedure_revised = row['is_grouper_relevant_procedures_revised']

        add_procedure_true = [proc for proc in diff_procedures if proc not in original_procedures]

        if len(diff_procedures) != len(add_procedure_true):
            case_id = row['case_id']
            logger.info(f'Case {case_id} has deleted procedure')

        idx_add_procedures = [revised_procedures.index(added_proc) for added_proc in add_procedure_true]

        row['added_procedures'] = add_procedure_true
        row['added_procedure_grouper_relevant'] = [is_grouper_relevant_procedure_revised[idx] for idx in idx_add_procedures]

        return row

    logger.info('Extracting the list of added diagnosis and procedure codes ...')
    df = df.apply(_extract_revision, axis=1)
    # Analyse if the added procedure grouper relevant or not
    add_chops = df['added_procedures'].values
    add_chops_grouper = df['added_procedure_grouper_relevant'].values
    idx_add_chops_true = [i for i in range(0, len(add_chops)) if len(add_chops[i]) > 0]
    add_chops_true = add_chops[idx_add_chops_true]
    add_chops_true_grouper = add_chops_grouper[idx_add_chops_true]
    add_chops_true_unpacked = list(itertools.chain(*add_chops_true))
    add_chops_true_grouper_unpacked = list(itertools.chain(*add_chops_true_grouper))

    df_chop_drg_relevant = pd.DataFrame([add_chops_true_unpacked, add_chops_true_grouper_unpacked]).transpose()
    df_chop_drg_relevant.columns = ['added_procedure', 'is_grouper_relevant']
    df_chop_drg_relevant_count = df_chop_drg_relevant.groupby('added_procedure')['is_grouper_relevant'].value_counts().unstack(fill_value=0).reset_index()

    # save csv with all info : can be used for analysis drg relevant codes before or after revision
    output_path_revised_all_info = os.path.join(ROOT_DIR, 'resources', 'db_data', 'revised_case_all_info.csv')
    output_path_revised_drg_relevance = os.path.join(ROOT_DIR, 'resources', 'db_data', 'added_procedure_drg_relevance.csv')
    os.makedirs(os.path.dirname(output_path_revised_all_info), exist_ok=True)
    logger.info(f"Storing {df.shape[0]} cases at {output_path_revised_all_info} ...")
    df.to_csv(output_path_revised_all_info, index=False)
    df_chop_drg_relevant_count.to_csv(output_path_revised_drg_relevance, index=False)

    df.drop(columns=['reviewed_original', 'revised_original', 'revision_date_original', 'reviewed_revised',
                     'revised_revised', 'revision_date_revised', 'revision_id_original', 'revision_id_revised',
                     'pd_original', 'secondary_diagnoses_original', 'pd_revised', 'secondary_diagnoses_revised',
                     'primary_procedure_original', 'secondary_procedures_original', 'primary_procedure_revised',
                     'secondary_procedures_revised',
                     'adrg_original', 'adrg_revised',
                     ], inplace=True)

    output_path = os.path.join(ROOT_DIR, 'resources', 'db_data', 'revisions.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info(f"Storing {df.shape[0]} cases at {output_path} ...")
    df.to_csv(output_path, index=False)

    logger.success('done')


if __name__ == '__main__':
    extract_revisions()
