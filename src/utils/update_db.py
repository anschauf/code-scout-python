import pandas as pd
from beartype import beartype

from src.service.bfs_cases_db_service import insert_revised_cases_into_diagnoses, insert_revised_cases_into_procedures, \
    insert_revised_cases_into_revisions
from src.service.database import Database


@beartype
def insert_revised_cases_into_db(revised_cases: pd.DataFrame, diagnoses_df: pd.DataFrame, procedures_df: pd.DataFrame):
    """
    Insert revised cases into revisions, diagonoses and procedures tables
    @param revised_cases: a Dataframe of revised cases after grouping
    @param diagnoses_df: a Dataframe of revised cases for diagnoses after grouping
    @param procedures_df: a Dataframe of revised cases for procedures after grouping
    @return:
    """
    with Database() as db:
        sociodemographic_id_with_revision_id = insert_revised_cases_into_revisions(revised_cases, db.session)
        insert_revised_cases_into_diagnoses(diagnoses_df, sociodemographic_id_with_revision_id, db.session)
        insert_revised_cases_into_procedures(procedures_df, sociodemographic_id_with_revision_id, db.session)
