from unittest import TestCase

import pandas as pd

from src.models.clinic import Clinic
from src.models.diagnosis import Diagnosis
from src.models.duration_of_stay import DurationOfStay
from src.models.hospital import Hospital
from src.models.procedure import Procedure
from src.models.revision import Revision
from src.models.sociodemographics import Sociodemographics
from src.service.bfs_cases_db_service import get_all_revised_cases, get_sociodemographics_by_sociodemographics_ids, \
    get_original_revision_id_for_sociodemographic_ids, get_clinics, sociodemographics_revised_cases, \
    get_diagnoses_codes, get_procedures_codes, get_codes, get_all_revised_cases_before_revision, \
    query_revised_case_before_revision, get_original_revision_for_revision_ids
from src.service.database import Database
from src.models.sociodemographics import Sociodemographics, SOCIODEMOGRAPHIC_ID_COL

from src.utils.global_configs import *

class TestDbAccess(TestCase):
    def _read_one_row(self, table, session):
        query = session.query(table).limit(1)
        df = pd.read_sql(query.statement, session.bind)
        return df

    def test_access_to_db(self):
        with Database() as db:
            clinic_df = self._read_one_row(Clinic, db.session)
            diagnoses_df = self._read_one_row(Diagnosis, db.session)
            dos_df = self._read_one_row(DurationOfStay, db.session)
            hospital_df = self._read_one_row(Hospital, db.session)
            procedures_df = self._read_one_row(Procedure, db.session)
            revision_df = self._read_one_row(Revision, db.session)
            socio_df = self._read_one_row(Sociodemographics, db.session)

    def test_sociodemographics_revised_cases(self):
        with Database() as db:
            revised_cases_all = get_all_revised_cases(db.session)
            revised_case_sociodemographic_ids = revised_cases_all['sociodemographic_id'].values.tolist()
            sociodemographics_revised_cases = get_sociodemographics_by_sociodemographics_ids(
                revised_case_sociodemographic_ids, db.session)

            self.assertEqual(sociodemographics_revised_cases.shape[0], revised_cases_all.shape[0])

    def test_revised_case_with_codes_after_revision(self):
        with Database() as db:
            revised_case_all = get_all_revised_cases(db.session)
            df_diagnoses = get_diagnoses_codes(revised_case_all, db.session)
            df_procedures = get_procedures_codes(revised_case_all, db.session)

            #  reset index as revision_id
        revised_case_all.set_index(REVISION_ID_COL, inplace=True)
        df_diagnoses.set_index(REVISION_ID_COL, inplace=True)
        df_procedures.set_index(REVISION_ID_COL, inplace=True)

        #  merge all data using revision_id

        revised_cases_df = pd.concat([revised_case_all, df_diagnoses[[PRIMARY_DIAGNOSIS_COL, SECONDARY_DIAGNOSES_COL]],
                                      df_procedures[[PRIMARY_PROCEDURE_COL, SECONDARY_PROCEDURES_COL]]], axis=1)
        revised_cases_df.rename(columns={'old_pd': 'pd'}, inplace=True)
        revised_cases_df.to_csv('all_revised_cases.csv')
        print('')

    def test_all_revised_case_with_codes_before_revision(self):
        with Database() as db:
            revised_cases_all = get_all_revised_cases(db.session)
            revised_case_sociodemographic_ids = revised_cases_all[SOCIODEMOGRAPHIC_ID_COL].values.tolist()
            df = get_original_revision_id_for_sociodemographic_ids(revised_case_sociodemographic_ids, db.session)
            revision_id = df[REVISION_ID_COL].values.tolist()

            revised_case_orig = get_original_revision_for_revision_ids(revision_id, db.session)
            df_diagnoses = get_diagnoses_codes(revised_case_orig, db.session)
            df_procedures = get_procedures_codes(revised_case_orig, db.session)

            #  reset index as revision_id
        revised_case_orig.set_index(REVISION_ID_COL, inplace=True)
        df_diagnoses.set_index(REVISION_ID_COL, inplace=True)
        df_procedures.set_index(REVISION_ID_COL, inplace=True)

        #  merge all data using revision_id

        revised_cases_before_revision_all = pd.concat([revised_case_orig, df_diagnoses[[PRIMARY_DIAGNOSIS_COL, SECONDARY_DIAGNOSES_COL]],
                                      df_procedures[[PRIMARY_PROCEDURE_COL, SECONDARY_PROCEDURES_COL]]], axis=1)
        revised_cases_before_revision_all.rename(columns={'old_pd': 'pd'}, inplace=True)
        revised_cases_before_revision_all.to_csv('all_revised_cases_before_revision.csv')
        print('')

    def test_sociodemographics_revised_cases(self):
        with Database() as db:
            df = sociodemographics_revised_cases(db.session)
            df.reset_index(SOCIODEMOGRAPHIC_ID_COL, inplace=True)
            df.to_csv('sociodemographics_for_revised_cases.csv')
        print('')
        def test_get_clinics(self):

            with Database() as db:
                df = get_clinics(db.session)
        self.assertEqual(df.shape[0], 17)
        self.assertEqual(df.shape[1], 3)


# get revised case
# get revised case revision id
#  from revision id get icds
# from revision id get chops
# get original revision id



