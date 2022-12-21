import itertools
from os import listdir
from os.path import join

import numpy as np
import pandas as pd
from loguru import logger

from src import ROOT_DIR

dir_data = join(ROOT_DIR, 'resources', 'data')
all_files = [x for x in listdir(dir_data) if x.endswith('.json')]

all_pccl_diagnoses = list()
for idx, file in enumerate(all_files):
    logger.info(f'{(idx+1)}/{len(all_files)}: Reading {file}')
    df = pd.read_json(path_or_buf=join(dir_data, file), lines=True, dtype={'diagnosesForPccl': object})
    pccl_diagnoses = df['diagnosesForPccl'].values.tolist()
    pccl_diagnoses = [d for d in pccl_diagnoses if len(d) > 0]
    pccl_diagnoses = [list(d.keys()) for d in pccl_diagnoses]
    pccl_diagnoses = list(itertools.chain.from_iterable(pccl_diagnoses))
    all_pccl_diagnoses.extend(pccl_diagnoses)

pccl_diagnoses_df = pd.DataFrame(np.array(all_pccl_diagnoses).reshape(-1, 1), columns=['icd']).value_counts()
print(pccl_diagnoses_df.head(100))
print('')