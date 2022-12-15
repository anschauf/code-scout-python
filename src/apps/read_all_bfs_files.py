import os
import shutil
import subprocess
from pathlib import Path

import awswrangler.s3 as wr
import pandas as pd
# noinspection PyPackageRequirements
from decouple import config
from loguru import logger

from src import ROOT_DIR
# noinspection PyProtectedMember
from src.service.aimedic_grouper import _escape_ansi

JAR_FILE_PATH = f'{ROOT_DIR}/resources/jars/aimedic-grouper-assembly.jar'
AIMEDIC_GROUPER_CLASS = 'ch.aimedic.grouper.apps.AimedicGrouperApp'


def read_all_bfs_files(*,
                       s3_bucket: str,
                       bfs_files_path: str,
                       output_dir: str
                       ):
    root_path = f's3://{s3_bucket}/{bfs_files_path}/'

    # noinspection PyArgumentList
    bfs_filenames = wr.list_objects(root_path, suffix='.dat')
    logger.info(f"Found {len(bfs_filenames)} files at '{root_path}' ...")

    bfs_filenames_per_hospital = dict()
    for bfs_filename in bfs_filenames:
        filename = os.path.basename(bfs_filename)
        hospital_abbreviation = filename.split('_')[0]

        if hospital_abbreviation not in bfs_filenames_per_hospital:
            bfs_filenames_per_hospital[hospital_abbreviation] = list()

        bfs_filenames_per_hospital[hospital_abbreviation].append(bfs_filename)

    # Clear output folder
    shutil.rmtree(output_dir, ignore_errors=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Append the AWS credentials to the environment variables, so that the grouper can read files from S3. We copy all
    # the variables because we need to know JAVA_HOME to be able to run java
    env_vars = dict(os.environ)
    env_vars['AWS_ACCESS_KEY_ID'] = config('AWS_ACCESS_KEY_ID')
    env_vars['AWS_SECRET_ACCESS_KEY'] = config('AWS_SECRET_ACCESS_KEY')

    num_hospitals = len(bfs_filenames_per_hospital)

    for idx, (hospital_abbreviation, bfs_filenames) in enumerate(bfs_filenames_per_hospital.items()):
        logger.info(f"{idx + 1}/{num_hospitals}: Reading {len(bfs_filenames)} files for the hospital '{hospital_abbreviation}' ...")
        # Build a comma-separated list of filenames
        cs_bfs_filenames = ','.join(bfs_filenames)

        # Parse the files
        raw_output = subprocess.check_output([
            'java', '-cp', JAR_FILE_PATH, AIMEDIC_GROUPER_CLASS,
            'bfs-file', "--input", cs_bfs_filenames,
            "--all-vars"
        ], env=env_vars)

        # Read the data from the JSON format into a DataFrame
        output = _escape_ansi(raw_output.decode('UTF-8'))
        lines = output.split('\n')
        output_lines = [line for line in lines if line.startswith('{"')]

        df = pd.read_json('\n'.join(output_lines), orient='records', typ='frame', lines=True)

        # Export the data to JSON once more, because it will preserve nested fields in some columns
        output_path = os.path.join(output_dir, f'{hospital_abbreviation}.json')
        logger.info(f"Exporting a pandas.DataFrame with {df.shape[0]} cases to '{output_path}' ...")
        df.to_json(output_path, orient='records', lines=True)

    logger.success('done')


if __name__ == '__main__':
    read_all_bfs_files(
        s3_bucket='aimedic-patient-data',
        bfs_files_path='bfs',
        output_dir=f'{ROOT_DIR}/resources/data/'
    )
