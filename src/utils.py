from typing import Optional

import numpy as np
from numpy._typing import ArrayLike


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