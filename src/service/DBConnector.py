import boto3
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker


class DBConnector(object):
    """
    Template inspired from Stackoverflow:
    https://stackoverflow.com/questions/40525545/single-database-connection-throughout-the-python-application-following-singleto
    """

    def __init__(self, host: str, user: str, port: str, database: str, aws_region: str):
        self.host = host
        self.user = user
        self.port = port
        self.database = database
        self.aws_region = aws_region
        self.db_session = None

    def create_session(self):
        """
        Creates new DB connection
        @return:
        """
        # gets the credentials from .aws/credentials
        client = boto3.client('rds')

        engine = create_engine(f'postgresql://{self.user}@{self.host}:{self.port}/{self.database}')

        @event.listens_for(engine, "do_connect")
        def receive_do_connect(dialect, conn_rec, cargs, cparams):
            token = client.generate_db_auth_token(DBHostname=self.host, Port=self.port,
                                                  DBUsername=self.user, Region=self.aws_region)
            cparams["password"] = token

        Session = sessionmaker(bind=engine)
        return Session()

    # For explicitly opening database connection
    def __enter__(self):
        self.db_session = self.create_session()
        return self.db_session

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.db_session
