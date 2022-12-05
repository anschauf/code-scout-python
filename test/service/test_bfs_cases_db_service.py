from unittest import TestCase

import pandas as pd

from src.models.clinic import Clinic
from src.models.diagnosis import Diagnosis
from src.models.duration_of_stay import DurationOfStay
from src.models.hospital import Hospital
from src.models.procedure import Procedure
from src.models.revision import Revision
from src.models.sociodemographics import Sociodemographics
from src.service.database import Database


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
