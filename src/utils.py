import io
import os

import boto3
import numpy as np
from loguru import logger
from matplotlib import pyplot

s3_prefix = 's3://'

def get_categorical_ranks(non_categorical_rank, label_not_suggested: str = 'not suggested'):
    categorical_rank = np.zeros((5,))
    if non_categorical_rank == label_not_suggested:
        categorical_rank[-1] = 1
    elif non_categorical_rank in np.arange(1, 4):
        categorical_rank[0] = 1
    elif non_categorical_rank in np.arange(4, 7):
        categorical_rank[1] = 1
    elif non_categorical_rank in np.arange(7, 10):
        categorical_rank[2] = 1
    else:
        categorical_rank[3] = 1
    return categorical_rank


def save_figure_to_pdf_on_s3(plt: pyplot, bucket: str, filename: str):
    filename = __remove_prefix_and_bucket_if_exists(filename)
    img_data = io.BytesIO()
    plt.savefig(img_data, format='pdf', bbox_inches='tight')
    img_data.seek(0)
    s3 = boto3.resource('s3')
    bucket_s3 = s3.Bucket(bucket)
    bucket_s3.upload_fileobj(img_data, filename)


def __remove_prefix_and_bucket_if_exists(filename: str) -> str:
    """Remove s3 prefix and the bucket folder if filename starts with s3 prefix.

    @param filename: A filename.
    @return: The filname without s3 prefix and the bucket.
    """
    if filename.startswith(s3_prefix):
        filename_without_prefix = filename.replace(s3_prefix, '')
        filename_without_prefix_split = filename_without_prefix.split('/')
        logger.warning(f'removed prefix {s3_prefix} and the bucket folder {filename_without_prefix_split[0]} from the path')
        return os.path.join(*np.asarray(filename_without_prefix_split)[1:])
    else:
        return filename