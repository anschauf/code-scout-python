from src.service.DBConnector import DBConnector
import psycopg2
from decouple import config
import pandas as pd

BFS_CASES_DB_URL = config('BFS_CASES_DB_URL')
BFS_CASES_DB_USER = config('BFS_CASES_DB_USER')
BFS_CASES_DB_NAME = config('BFS_CASES_DB_NAME')
BFS_CASES_DB_PORT = config('BFS_CASES_DB_PORT')
AWS_REGION = config('AWS_REGION')


class DBConnection(object):
    session = None

    @classmethod
    def get_connection(cls, new=False, host=BFS_CASES_DB_URL, user=BFS_CASES_DB_USER, database=BFS_CASES_DB_NAME,
                       port=BFS_CASES_DB_PORT, aws_region=AWS_REGION):
        """Creates return new Singleton database connection"""
        if new or not cls.session:
            cls.connection = DBConnector(
                host, user, database, port, aws_region
            ).create_session()
        return cls.session

    @classmethod
    def execute_query(cls, query):
        """execute query on singleton db connection"""
        session = cls.get_connection()
        try:
            return pd.read_sql(query, session.bind())
        except psycopg2.ProgrammingError:
            session = cls.get_connection(new=True)  # Create new connection

        return pd.read_sql(query, session.bind())
