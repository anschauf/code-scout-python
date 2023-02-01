import os.path
import os.path
import re
import subprocess

# noinspection PyPackageRequirements
import pandas as pd
from beartype import beartype
from decouple import config

from src import ROOT_DIR


class AimedicGrouper:
    def __init__(self):
        self.jar_file_path = f'{ROOT_DIR}/resources/jars/aimedic-grouper-assembly.jar'
        if not os.path.exists(self.jar_file_path):
            raise IOError(f"The aimedic-grouper JAR file is not available at '{self.jar_file_path}")

        is_java_running = subprocess.check_call(['java', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if is_java_running != 0:
            raise Exception('Java is not accessible')

        # Append the AWS credentials to the environment variables, so that the grouper can read files from S3. We copy all
        # the variables because we need to know JAVA_HOME to be able to run java
        self.env_vars = dict(os.environ)  # TODO Copy only JAVA_HOME instead of passing all the vars
        self.env_vars['AWS_ACCESS_KEY_ID'] = config('AWS_ACCESS_KEY_ID')
        self.env_vars['AWS_SECRET_ACCESS_KEY'] = config('AWS_SECRET_ACCESS_KEY')

        self._ansi_escape = re.compile(r'(?:\x9B|\x1B\[)[0-?]*[ -/]*[@-~]')

    def _run_java_grouper_and_collect_output(self, mode: str, grouper_input: str | list[str]) -> pd.DataFrame:
        if isinstance(grouper_input, list):
            grouper_input = ','.join(grouper_input)

        raw_output = subprocess.check_output([
            'java', '-cp', self.jar_file_path, 'ch.aimedic.grouper.apps.AimedicGrouperApp',
            mode, '--input', grouper_input,
            '--all-vars'
        ], env=self.env_vars)

        # Read the data from the JSON format into a DataFrame
        output = self._ansi_escape.sub('', raw_output.decode('UTF-8'))
        lines = output.split('\n')
        output_lines = [line for line in lines if line.startswith('{"')]
        df = pd.read_json('\n'.join(output_lines), orient='records', typ='frame', lines=True)
        return df

    @beartype
    def run_batch_grouper(self, cases: str | list[str]) -> pd.DataFrame:
        if len(cases) == 0:
            raise ValueError('You must pass some cases to group')

        return self._run_java_grouper_and_collect_output('batch-grouper-string', cases)

    @beartype
    def run_bfs_file_parser(self, bfs_filenames: str | list[str]) -> pd.DataFrame:
        if len(bfs_filenames) == 0:
            raise ValueError('You must pass some BfS files to parse')

        return self._run_java_grouper_and_collect_output('bfs-file', bfs_filenames)


AIMEDIC_GROUPER = AimedicGrouper()
