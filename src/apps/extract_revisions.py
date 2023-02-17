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

        add_diagnoses = tuple(set(revised_diagnoses).difference(set(original_diagnoses)))
        row['added_diagnoses'] = add_diagnoses

        idx_add_diagnoses = [revised_diagnoses.index(added_diag) for added_diag in add_diagnoses]

        is_grouper_relevant_diagnoses_revised = row['is_grouper_relevant_diagnoses_revised']
        assert len(is_grouper_relevant_diagnoses_revised) == len(revised_diagnoses), 'Case has deleted diagnoses'
        row['added_diagnoses_grouper_relevant'] = set(
            [is_grouper_relevant_diagnoses_revised[idx] for idx in idx_add_diagnoses])

        original_procedures = row['all_procedures_original']
        revised_procedures = row['all_procedures_revised']

        original_procedures = [p.split(':')[0] for p in original_procedures]
        original_procedures = [p for p in original_procedures if p != '' and p != 'nan']

        revised_procedures = [p.split(':')[0] for p in revised_procedures]
        revised_procedures = [p for p in revised_procedures if p != '' and p != 'nan']

        added_procedures = tuple(set(revised_procedures).difference(set(original_procedures)))
        row['added_procedures'] = added_procedures
        idx_add_procedures = [revised_procedures.index(added_proc) for added_proc in added_procedures]
        is_grouper_relevant_procedure_revised = row['is_grouper_relevant_procedures_revised']
        assert len(is_grouper_relevant_diagnoses_revised) == len(revised_diagnoses), 'Case has deleted procedures'
        row['added_procedure_grouper_relevant'] = set(
            [is_grouper_relevant_procedure_revised[idx] for idx in idx_add_procedures])

        return row

    logger.info('Extracting the list of added diagnosis and procedure codes ...')
    df = df.apply(_extract_revision, axis=1)

    # save csv with all info : can be used for analysis drg relevant codes before or after revision
    output_path_revised_all_info = os.path.join(ROOT_DIR, 'resources', 'db_data', 'revised_case_all_info.csv')
    os.makedirs(os.path.dirname(output_path_revised_all_info), exist_ok=True)
    logger.info(f"Storing {df.shape[0]} cases at {output_path_revised_all_info} ...")
    df.to_csv(output_path_revised_all_info, index=False)

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
