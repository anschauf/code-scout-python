import os
import shutil
from pathlib import Path

import awswrangler.s3 as wr
# noinspection PyPackageRequirements
from decouple import config
from loguru import logger

from src import ROOT_DIR
# noinspection PyProtectedMember
from src.service.aimedic_grouper import AIMEDIC_GROUPER


def read_all_bfs_files(*,
                       s3_bucket: str,
                       bfs_files_path: str,
                       output_dir: str
                       ):
    """Read a list of BfS files, export all the known variables from them, and store the results into a JSON file, which
    can be loaded into a pandas.DataFrame.

    @param s3_bucket: The bucket on S3, where to find the files.
    @param bfs_files_path: A root path where to find the files in the `s3_bucket`.
    @param output_dir: The local output directory where to store the intermediate files, and the final concatenated file.

    @note The `bfs_files_path` will be searched recursively in all its sub-folders.
    @note All BfS files are expected to have the following pattern: `<hospital-name>_<year>.dat`
    @note The files are processed one at a time, to avoid out-of-memory errors in the java subprocess. The files are
        not concatenated because otherwise pandas will run out of memory.
    @note A column containing the abbreviated name of the hospital is appended to each dataset.
    """
    root_path = f's3://{s3_bucket}/{bfs_files_path}/'

    # noinspection PyArgumentList
    bfs_filenames = wr.list_objects(root_path, suffix='.dat')
    num_files = len(bfs_filenames)
    logger.info(f"Found {num_files} files at '{root_path}'")

    # Clear the local results folder
    shutil.rmtree(output_dir, ignore_errors=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Append the AWS credentials to the environment variables, so that the grouper can read files from S3. We copy all
    # the variables because we need to know JAVA_HOME to be able to run java
    env_vars = dict(os.environ)  # TODO Copy only JAVA_HOME instead of passing all the vars
    env_vars['AWS_ACCESS_KEY_ID'] = config('AWS_ACCESS_KEY_ID')
    env_vars['AWS_SECRET_ACCESS_KEY'] = config('AWS_SECRET_ACCESS_KEY')

    for idx, bfs_filename in enumerate(bfs_filenames):
        logger.info(f"{idx + 1}/{num_files}: Reading '{bfs_filename}' ...")
        df = AIMEDIC_GROUPER.run_bfs_file_parser(bfs_filename)

        # Append the name of the hospital
        filename = os.path.basename(bfs_filename)
        hospital_abbreviation = filename.split('_')[0]
        df['hospital'] = hospital_abbreviation

        # Export the data to JSON once more (instead of, e.g., CSV), because it will preserve any nested fields
        filename_wo_ext = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f'{filename_wo_ext}.json')
        logger.info(f"Writing {df.shape[0]} cases to '{output_path}' ...")
        df.to_json(output_path, orient='records', lines=True)

    logger.success('done')


if __name__ == '__main__':
    read_all_bfs_files(
        s3_bucket='aimedic-patient-data',
        bfs_files_path='bfs',
        output_dir=f'{ROOT_DIR}/resources/data/'
    )
