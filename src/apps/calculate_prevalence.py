import pandas as pd
from loguru import logger

from src.models.diagnosis import Diagnosis
from src.models.procedure import Procedure
from src.service.database import Database


def calculate_prevalence():
    with Database() as db:
        logger.info('Reading the diagnoses ...')
        query_diagnoses = (
            db.session
            .query(Diagnosis)
            .with_entities(Diagnosis.code, Diagnosis.is_primary)
            .filter(Diagnosis.is_grouper_relevant)
        )
        diagnoses_df = pd.read_sql(query_diagnoses.statement, db.session.bind)

        logger.info('Reading the procedures ...')
        query_procedures = (
            db.session
            .query(Procedure)
            .with_entities(Procedure.code, Procedure.is_primary)
            .filter(Procedure.is_grouper_relevant)
        )
        procedures_df = pd.read_sql(query_procedures.statement, db.session.bind)

    logger.info('Aggregating and calculating the prevalence ...')
    diagnosis_prevalence_df = diagnoses_df['code'].value_counts()
    procedure_prevalence_df = procedures_df['code'].value_counts()

    print('')

    #
    #
    # df = pd.merge(original_cases, revised_cases, on='sociodemographic_id', suffixes=('_original', '_revised'))
    # df = pd.merge(df, sociodemographics, on='sociodemographic_id')
    # df = df.sort_values(by='sociodemographic_id', ascending=True).reset_index(drop=True)
    #
    # sorted_column_pairs = ['sociodemographic_id', 'case_id', 'hospital']
    # for col_name in sorted_columns[1:]:
    #     sorted_column_pairs.append(f'{col_name}_original')
    #     sorted_column_pairs.append(f'{col_name}_revised')
    # df = df[sorted_column_pairs]
    #
    # def _extract_revision(row):
    #     original_diagnoses = [row['pd_original']]
    #     original_diagnoses.extend(sorted(row['secondary_diagnoses_original']))
    #     revised_diagnoses = [row['pd_revised']]
    #     revised_diagnoses.extend(sorted(row['secondary_diagnoses_revised']))
    #
    #     row['original_diagnoses'] = original_diagnoses
    #     row['revised_diagnoses'] = revised_diagnoses
    #     row['added_diagnoses'] = tuple(set(revised_diagnoses).difference(original_diagnoses))
    #
    #     original_procedures = [row['primary_procedure_original']]
    #     original_procedures.extend(row['secondary_procedures_original'])
    #     revised_procedures = [row['primary_procedure_revised']]
    #     revised_procedures.extend(row['secondary_procedures_revised'])
    #
    #     original_procedures = [p.split(':')[0] for p in original_procedures]
    #     original_procedures = [p for p in original_procedures if p != '' and p != 'nan']
    #
    #     revised_procedures = [p.split(':')[0] for p in revised_procedures]
    #     revised_procedures = [p for p in revised_procedures if p != '' and p != 'nan']
    #
    #     row['original_procedures'] = sorted(original_procedures)
    #     row['revised_procedures'] = sorted(revised_procedures)
    #     row['added_procedures'] = tuple(set(revised_procedures).difference(original_procedures))
    #
    #     return row
    #
    # logger.info('Extracting the list of added diagnosis and procedure codes ...')
    # df = df.apply(_extract_revision, axis=1)
    #
    # df.drop(columns=['reviewed_original', 'revised_original', 'revision_date_original', 'reviewed_revised', 'revised_revised', 'revision_date_revised', 'revision_id_original', 'revision_id_revised',
    #                  'pd_original', 'secondary_diagnoses_original', 'pd_revised', 'secondary_diagnoses_revised',
    #                  'primary_procedure_original', 'secondary_procedures_original', 'primary_procedure_revised', 'secondary_procedures_revised',
    #                  'adrg_original', 'adrg_revised',
    #                  ], inplace=True)
    #
    # output_path = os.path.join(ROOT_DIR, 'resources', 'db_data', 'revisions.csv')
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # logger.info(f"Storing {df.shape[0]} cases at {output_path} ...")
    # df.to_csv(output_path, index=False)

    logger.success('done')


if __name__ == '__main__':
    calculate_prevalence()
