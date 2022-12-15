import os
import re
import subprocess

import pandas as pd
from decouple import config

from src import ROOT_DIR

jar_file_path = f'{ROOT_DIR}/resources/jars/aimedic-grouper-assembly.jar'


def _escape_ansi(line):
    ansi_escape = re.compile(r'(?:\x9B|\x1B\[)[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', line)


# grouper_format = "41282182;67;0;;W;20170216;01;20170223;0;7;0;I313|J91|I318|E788|I1090|J9580;371211:L:20170216|340999:L:20170216|009910::20170216|3491:L:20170219;"

AWS_ACCESS_KEY_ID = config('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = config('AWS_SECRET_ACCESS_KEY')


env_vars = dict(os.environ)
env_vars['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
env_vars['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY



raw_output = subprocess.check_output([
    'java', '-cp', jar_file_path, 'ch.aimedic.grouper.apps.AimedicGrouperApp',
    'bfs-file', "--input", "s3://aimedic-patient-data/bfs/hirslanden/andreasklinik_cham_zug/AK_2017.dat,s3://aimedic-patient-data/bfs/hirslanden/andreasklinik_cham_zug/AK_2018.dat",
    "--all-vars"
], env=env_vars, shell=True)

output = _escape_ansi(raw_output.decode('UTF-8'))
lines = output.split('\n')
output_lines = [line for line in lines if line.startswith('{"')]

df = pd.read_json('\n'.join(output_lines), orient='records', typ='frame', lines=True, dtype={'birthDate': str, 'GeburtsdatumDerMutter': str})
print(df)

print("")
