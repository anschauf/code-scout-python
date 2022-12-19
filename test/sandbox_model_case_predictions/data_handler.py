from os import listdir
from os.path import join

import pandas as pd
from loguru import logger

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
    'pccl': 'string',
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
    'AlterDerMutter': 'string'
}


def load_data(dir_data = join(ROOT_DIR, 'resources', 'data')):
    all_files = [x for x in listdir(dir_data) if x.endswith('.json')][:2]
    all_dfs = list()
    for idx, file in enumerate(all_files):
        logger.info(f'{(idx+1)}/{len(all_files)}: Reading {file}')
        all_dfs.append(pd.read_json(path_or_buf=join(dir_data, file), lines=True, dtype=DTYPES))

    all_data = pd.concat(all_dfs, ignore_index=True)
    return __preprocess_data(all_data)


def __preprocess_data(df: pd.DataFrame):
    # remove decimals
    df['birthDate'] = df['birthDate'].apply(lambda x: x.replace('.0', '') if isinstance(x, str) else x)
    df['GeburtsdatumDerMutter'] = df['GeburtsdatumDerMutter'].apply(lambda x: x.replace('.0', '') if isinstance(x, str) else x)
    df['KindVitalstatus'] = df['KindVitalstatus'].apply(lambda x: x.replace('.0', '') if isinstance(x, str) else x)
    df['KongenitaleMissbildungen'] = df['KongenitaleMissbildungen'].apply(lambda x: x.replace('.0', '') if isinstance(x, str) else x)
    df['Mehrlingsgeburten'] = df['Mehrlingsgeburten'].apply(lambda x: x.replace('.0', '') if isinstance(x, str) else x)
    df['AlterDerMutter'] = df['AlterDerMutter'].apply(lambda x: x.replace('.0', '') if isinstance(x, str) else x)

    return df
