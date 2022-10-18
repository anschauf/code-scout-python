import boto3
import pandas as pd
from decouple import config
from pandas import DataFrame
from sqlalchemy import create_engine, event, extract, func
from sqlalchemy.orm import sessionmaker

from src.models.BfsCase import BfsCase
from src.models.Clinic import Clinic
from src.models.Hospital import Hospital
from src.models.chop_code import ChopCode
from src.models.icd_code import IcdCode

BFS_CASES_DB_URL = config('BFS_CASES_DB_URL')
BFS_CASES_DB_USER = config('BFS_CASES_DB_USER')
BFS_CASES_DB_NAME = config('BFS_CASES_DB_NAME')
AWS_REGION = config('AWS_REGION')
BFS_CASES_DB_PORT = config('BFS_CASES_DB_PORT')
AWS_ACCESS_KEY_ID = config('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = config('AWS_SECRET_ACCESS_KEY')


class Database:
    """
    Bfs cases connection class.
    Based on model found here: https://stackoverflow.com/a/38078544

    Example usage:
        with Database() as db:
            df = db.get_hospital_year_cases('USZ', 2019)
    """

    def __init__(self,
                 region_name=AWS_REGION,
                 aws_access_key_id=AWS_ACCESS_KEY_ID,
                 aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                 db_host=BFS_CASES_DB_URL,
                 db_user=BFS_CASES_DB_USER,
                 db_name=BFS_CASES_DB_NAME,
                 port=BFS_CASES_DB_PORT
                 ):
        self._client = boto3.client('rds', region_name=region_name, aws_access_key_id=aws_access_key_id,
                                    aws_secret_access_key=aws_secret_access_key)
        engine = create_engine(
            f'postgresql://{db_user}@{db_host}:{port}/{db_name}')

        @event.listens_for(engine, "do_connect")
        def receive_do_connect(dialect, conn_rec, cargs, cparams):
            token = self._client.generate_db_auth_token(DBHostname=db_host, Port=port,
                                                        DBUsername=db_user, Region=region_name)
            cparams["password"] = token

        # create a configured "Session" class
        Session = sessionmaker(bind=engine)
        self._session = Session()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def connection(self):
        return self._session

    def commit(self):
        self._session.commit()

    def close(self, commit=True):
        if commit:
            self.commit()
        self._client.close()

    def execute(self, sql, params=None):
        self._session.execute(sql, params or ())

    def query(self, sql_query):
        return pd.read_sql(sql_query, self._session.bind)



    def get_hospital_cases_df(self, hopsital_name) -> DataFrame:
        """

        @param hopsital_name:

        @return:
        """
        query = self._session.query(BfsCase).join(Hospital).filter(Hospital.name == hopsital_name)
        return pd.read_sql(query.statement, self._session.bind)

    def get_clinics(self):
        return self._session.query(Clinic).all()


    def get_hospital_year_cases(self, hospital_name, year):
        """
        Get the cases filtered by year and hospital name, joint together with all its ICD and CHOP codes.
        @param hospital_name:
        @param year:
        @return: a Dataframe with all matching cases.
        """
        subquery_cases_from_hospital_year = self._session.query(BfsCase).join(Hospital).filter(Hospital.name == hospital_name).filter(extract('year', BfsCase.discharge_date) == year).subquery()

        subquery_icds = self._session.query(IcdCode.aimedic_id,
                                            func.array_agg(IcdCode.code).label('icds'),
                                            func.array_agg(IcdCode.ccl).label('icds_ccl'),
                                            func.array_agg(IcdCode.is_primary).label('icds_is_primary'),
                                            func.array_agg(IcdCode.is_grouper_relevant).label('icds_is_grouper_relevant')
                                            ).group_by(IcdCode.aimedic_id).subquery()
        subquery_chops = self._session.query(ChopCode.aimedic_id,
                                             func.array_agg(ChopCode.code).label('chops'),
                                             func.array_agg(ChopCode.side).label('chops_side'),
                                             func.array_agg(ChopCode.date).label('chops_date'),
                                             func.array_agg(ChopCode.is_grouper_relevant).label('chops_is_grouper_relevant'),
                                             func.array_agg(ChopCode.is_primary).label('chops_is_primary'),
                                             ).group_by(ChopCode.aimedic_id).subquery()

        subquery_bfs_icds = self._session.query(subquery_cases_from_hospital_year,
                                                subquery_icds.c.icds,
                                                subquery_icds.c.icds_ccl,
                                                subquery_icds.c.icds_is_primary,
                                                subquery_icds.c.icds_is_grouper_relevant
                                                ).join(subquery_icds, subquery_cases_from_hospital_year.c.aimedic_id == subquery_icds.c.aimedic_id, isouter=True).subquery()

        query = self._session.query(subquery_bfs_icds,
                                    subquery_chops.c.chops,
                                    subquery_chops.c.chops_side,
                                    subquery_chops.c.chops_date,
                                    subquery_chops.c.chops_is_grouper_relevant,
                                    subquery_chops.c.chops_is_primary
                                    ).join(subquery_chops, subquery_bfs_icds.c.aimedic_id == subquery_chops.c.aimedic_id, isouter=True)


        return pd.read_sql(query.statement, self._session.bind)
