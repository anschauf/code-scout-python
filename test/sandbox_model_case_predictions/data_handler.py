from os import listdir
from os.path import join
from typing import Optional

import pandas as pd
from beartype import beartype
from loguru import logger
from tqdm import tqdm

from src import ROOT_DIR

DTYPES = {
    'id': 'string',
    'entryDate': 'string',
    'exitDate': 'string',
    'birthDate': 'string', # first to int to get rid of decimals
    'leaveDays': int,
    'ageYears': int,
    'ageDays': int,
    'admissionWeight': int,
    'gender': 'string',
    'grouperAdmissionCode': 'string',
    'grouperDischargeCode': 'string',
    'gestationAge': int,
    'durationOfStay': int,
    'hoursMechanicalVentilation': int,
    'primaryDiagnosis': 'string',
    'secondaryDiagnoses': object,
    'procedures': object,
    'medications': object,
    'ageFlag': 'string',
    'admissionWeightFlag': 'string',
    'genderFlag': 'string',
    'grouperAdmissionCodeFlag': 'string',
    'grouperDischargeCodeFlag': 'string',
    'durationOfStayFlag': 'string',
    'hoursMechanicalVentilationFlag': 'string',
    'gestationAgeFlag': 'string',
    'isNewborn': bool,
    'durationOfStayCaseType': 'string',
    'grouperStatus': 'string',
    'drg': 'string',
    'mdc': 'string',
    'mdcPartition': 'string',
    'pccl': int,
    'rawPccl': float,
    'diagnosesForPccl': object,
    'drgRelevantDiagnoses': object,
    'drgRelevantProcedures': object,
    'diagnosesExtendedInfo': object,
    'proceduresExtendedInfo': object,
    'drgCostWeight': float,
    'effectiveCostWeight': float,
    'drgRelevantGlobalFunctions': object,
    'supplementCharges': float,
    'supplementChargePerCode': object,
    'AnonymerVerbindungskode': 'string',
    'ArtDesSGIScore': 'string',
    'AufenthaltIntensivstation': int,
    'AufenthaltNachAustritt': 'string',
    'AufenthaltsKlasse': 'string',
    'AufenthaltsortVorDemEintritt': 'string',
    'BehandlungNachAustritt': 'string',
    'Eintrittsart': 'string',
    'EinweisendeInstanz': 'string',
    'EntscheidFuerAustritt': 'string',
    'ErfassungDerAufwandpunkteFuerIMC': int,
    'Hauptkostenstelle': 'string',
    'HauptkostentraegerFuerGrundversicherungsleistungen': 'string',
    'NEMSTotalAllerSchichten': int,
    'WohnortRegion': 'string',
    'NumDrgRelevantDiagnoses': int,
    'NumDrgRelevantProcedures': int,
    'GeburtsdatumDerMutter': 'string',
    'KindVitalstatus': 'string',
    'KongenitaleMissbildungen': 'string',
    'Mehrlingsgeburten': 'string',
    'AlterDerMutter': 'string',
    'VectorizedCodes': object,
}


@beartype
def load_data(
        dir_data = join(ROOT_DIR, 'resources', 'data'),
        *,
        columns: Optional[list[str]] = None,
        only_2_rows: bool = False
        ):
    all_files = [x for x in listdir(dir_data) if x.endswith('.json')]

    if only_2_rows:
        all_data = pd.read_json(path_or_buf=join(dir_data, all_files[0]), lines=True, dtype=DTYPES).loc[:1] \
            .reset_index(drop=False)

    else:
        if columns is not None:
            logger.info(f'Reading {len(columns)} columns from {len(all_files)} files ...')
        else:
            logger.info(f'Reading all the columns from {len(all_files)} files ...')

        all_dfs = list()
        for file in tqdm(all_files):
            df = pd.read_json(path_or_buf=join(dir_data, file), lines=True, dtype=DTYPES)

            if columns is not None:
                df = df[columns]

            all_dfs.append(df)

        logger.info(f'Concatenating {len(all_dfs)} DataFrames ...')
        all_data = (
            pd.concat(all_dfs, ignore_index=True)
            # Add the `index` column to keep track of the original sorting
            .reset_index(drop=False)
        )
        logger.success(f'Loaded {all_data.shape[0]} rows')

    return __preprocess_data(all_data)


def __preprocess_data(df: pd.DataFrame):
    # Remove decimals
    for col_name in ('birthDate', 'GeburtsdatumDerMutter', 'KindVitalstatus', 'KongenitaleMissbildungen', 'Mehrlingsgeburten', 'AlterDerMutter'):
        if col_name in df.columns:
            df[col_name] = df[col_name].apply(lambda x: x.replace('.0', '') if isinstance(x, str) else x)

    # compute discharge year
    if 'exitDate' in df.columns:
        df['dischargeYear'] = df['exitDate'].apply(lambda x: x[:4] if isinstance(x, str) else '')

    return df
