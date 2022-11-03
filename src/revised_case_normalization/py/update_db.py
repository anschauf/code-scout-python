import pandas as pd
from beartype import beartype
from loguru import logger

from src.service.bfs_cases_db_service import insert_revised_cases_into_revisions, insert_revised_cases_into_diagnoses, \
    insert_revised_cases_into_procedures


@beartype
def update_db(revised_cases: pd.DataFrame, diagnoses_df: pd.DataFrame, procedures_df: pd.DataFrame):
    aimedic_id_with_revision_id = insert_revised_cases_into_revisions(revised_cases)
    insert_revised_cases_into_diagnoses(diagnoses_df, aimedic_id_with_revision_id)
    insert_revised_cases_into_procedures(procedures_df, aimedic_id_with_revision_id)
