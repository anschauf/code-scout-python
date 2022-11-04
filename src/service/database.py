import boto3
from decouple import config
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

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
        engine = create_engine(f'postgresql://{db_user}@{db_host}:{port}/{db_name}')

        @event.listens_for(engine, "do_connect")
        def receive_do_connect(dialect, conn_rec, cargs, cparams):
            token = self._client.generate_db_auth_token(DBHostname=db_host, Port=port,
                                                        DBUsername=db_user, Region=region_name)
            cparams["password"] = token

        self._session_class = sessionmaker(bind=engine)
        self.session = None

    def __enter__(self):
        self.session = self._session_class()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def connection(self):
        return self.session

    def commit(self):
        self.session.commit()

    def close(self, commit=True):
        if commit:
            self.commit()
        self.session.close()
        self.session = None

    def execute(self, sql, params=None):
        self.session.execute(sql, params or ())
