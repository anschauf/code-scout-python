import itertools
import math
import os
import re
from os import listdir
from os.path import join
from typing import Optional

import awswrangler.s3 as wr
import boto3
import numpy as np
import pandas as pd
import srsly
from beartype import beartype
from loguru import logger
from natsort import natsorted
from srsly.ujson import ujson
from tqdm import tqdm

from src import ROOT_DIR

tqdm.pandas()


DTYPES = {
    'id': 'string',
    'entryDate': 'string',
    'exitDate': 'string',
    'birthDate': 'string',  # first to int to get rid of decimals
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
        dir_data=join(ROOT_DIR, 'resources', 'data'),
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


@beartype
def load_data_single_file(
        dir_data=join(ROOT_DIR, 'resources', 'data'),
        *,
        columns: Optional[list[str]] = None,
        file_name='KSW_2020.json'):
    logger.info(f'Reading all the columns from {file_name} ...')

    df = pd.read_json(path_or_buf=join(dir_data, file_name), lines=True, dtype=DTYPES)

    logger.success(f'Loaded {df.shape[0]} rows')

    return __preprocess_data(df)


def __preprocess_data(df: pd.DataFrame):
    # Remove decimals
    for col_name in ('birthDate', 'GeburtsdatumDerMutter', 'KindVitalstatus', 'KongenitaleMissbildungen', 'Mehrlingsgeburten', 'AlterDerMutter'):
        if col_name in df.columns:
            df[col_name] = df[col_name].apply(lambda x: x.replace('.0', '') if isinstance(x, str) else x)

    # compute discharge year
    if 'exitDate' in df.columns:
        df['dischargeYear'] = df['exitDate'].apply(lambda x: x[:4] if isinstance(x, str) else '')

    return df


@beartype
def engineer_mind_bend_suggestions(
        revised_case_info_df: pd.DataFrame,
        files_path: str,
        ) -> pd.DataFrame:

    # Map each caseID to its original CW
    case_id_to_cw_map = dict()

    # noinspection PyUnresolvedReferences,PyProtectedMember
    def _get_old_cw(row):
        cw = row['cw_old']
        if isinstance(cw, pd._libs.missing.NAType):
            cw = 0.0
        else:
            cw = float(cw)

        if not np.isnan(cw):
            case_id_to_cw_map[row['id']] = cw
    revised_case_info_df.progress_apply(_get_old_cw, axis=1)

    # Map each caseID to its revisions
    case_id_to_revisions_map = dict()

    # noinspection PyShadowingNames
    def _get_revision(row):
        target_drg = row['drg_new']
        diagnoses_added = row['diagnoses_added']
        procedures_added = row['procedures_added']

        codes_added = set()
        # noinspection PyUnresolvedReferences,PyProtectedMember
        if not isinstance(diagnoses_added, pd._libs.missing.NAType):
            codes_added.update(diagnoses_added.split('|'))
        # noinspection PyUnresolvedReferences,PyProtectedMember
        if not isinstance(procedures_added, pd._libs.missing.NAType):
            codes_added.update(procedures_added.split('|'))
        case_id_to_revisions_map[row['id']] = (target_drg, codes_added)
    revised_case_info_df[revised_case_info_df['is_revised'] == 1].progress_apply(_get_revision, axis=1)

    revised_case_ids = set(revised_case_info_df[revised_case_info_df['is_revised'] == 1]['id'].values.tolist())

    is_files_path_on_s3 = files_path.startswith('s3://')

    if is_files_path_on_s3:
        all_files = wr.list_objects(files_path)
    else:
        all_files = os.listdir(files_path)
        all_files = [os.path.join(files_path, f) for f in all_files]
    all_files = [f for f in all_files if f.endswith('.json')]
    logger.info(f'Found {len(all_files)} files at {files_path=}')

    if is_files_path_on_s3:
        matches = list(re.compile('^s3://(.+?)/(.+)$').finditer(files_path))[0]
        bucket, prefix = matches.groups()
        s3 = boto3.client('s3')

    all_files = natsorted(all_files)

    all_data = list()

    for filename in tqdm(all_files):
        if is_files_path_on_s3:
            # noinspection PyUnboundLocalVariable
            key = prefix + filename.removeprefix(files_path)
            # noinspection PyUnboundLocalVariable
            json_text = s3.get_object(Bucket=bucket, Key=key)['Body'].read().decode('utf-8')
            json = ujson.loads(json_text)
        else:
            json = srsly.read_json(filename)

        feature_matrix = list()

        def __is_suggestion_a_complex_procedure(s):
            if 'globalFunctions' in s:
                return any('kompl' in gf.lower() for gf in s['globalFunctions'])
            else:
                return False

        for case_id, suggestions in json.items():
            case_id = case_id.lstrip('0')

            is_case_revised = case_id in revised_case_ids

            added_codes = set()
            does_contain_revised_drg = False
            n_revised_codes_in_suggestions = 0
            n_added_codes = 0

            if is_case_revised:
                revision = case_id_to_revisions_map.get(case_id, None)
                if revision is not None:
                    revised_drg, added_codes = revision
                    n_added_codes = len(added_codes)

                    does_contain_revised_drg = revised_drg in suggestions.keys()

                    all_suggested_codes = {suggestion['code'] for suggestion in itertools.chain.from_iterable(s for s in suggestions.values())}
                    n_revised_codes_in_suggestions = len(added_codes.intersection(all_suggested_codes))

            case_drg_cost_weight = case_id_to_cw_map.get(case_id, None)

            sum_p_suggestions_per_case = 0.0
            sum_support_suggestions_per_case = 0

            feature_rows_per_case = list()

            for target_drg, suggested_codes in suggestions.items():
                target_drg_cost_weight = suggested_codes[0]['targetDrgCostWeight']
                if case_drg_cost_weight is not None:
                    delta_target_drg_cost_weight = target_drg_cost_weight - case_drg_cost_weight
                else:
                    delta_target_drg_cost_weight = 0.0

                if is_case_revised:
                    all_suggested_codes = {s['code'] for s in suggested_codes}
                    n_added_codes_in_suggestions = len(added_codes.intersection(all_suggested_codes))
                else:
                    n_added_codes_in_suggestions = 0

                n_suggested_codes = len(suggested_codes)

                p_suggestions = np.sort([suggestion['p'] for suggestion in suggested_codes])[::-1]
                p_top_suggestion = p_suggestions[0]
                p_top3_suggestion = p_suggestions[:3].sum()
                p_top3_suggestion_norm = p_top3_suggestion / 3
                sum_p_suggestion = p_suggestions.sum()
                sum_p_suggestion_norm = sum_p_suggestion / p_suggestions.shape[0]
                sum_p_suggestions_per_case += sum_p_suggestion

                support_suggestions = np.array(sorted(list(itertools.chain.from_iterable([suggestion['support'] for suggestion in suggested_codes])))[::-1])
                support_top_suggestion = support_suggestions[0]
                support_top3_suggestion = support_suggestions[:3].sum()
                sum_support_suggestion = support_suggestions.sum()
                sum_support_suggestions_per_case += sum_support_suggestion

                suggestion_types = [suggestion['codeType'] for suggestion in suggested_codes]
                is_top_suggestion_a_chop = suggestion_types[0] == 'CHOP'
                is_top3_suggestion_a_chop = any(s == 'CHOP' for s in suggestion_types[:3])

                is_top_suggestion_a_complex_procedure = __is_suggestion_a_complex_procedure(suggested_codes[0])
                is_top3_suggestion_a_complex_procedure = any(__is_suggestion_a_complex_procedure(s) for s in suggested_codes[:3])

                p_entropy = np.sum(-p * math.log(p, 2) for p in p_suggestions)
                top3_p_entropy = np.sum(-p * math.log(p, 2) for p in p_suggestions[:3])

                feature_rows_per_case.append([case_id, is_case_revised,
                                              target_drg, target_drg_cost_weight, delta_target_drg_cost_weight,
                                              n_suggested_codes, n_added_codes_in_suggestions,
                                              p_top_suggestion, p_top3_suggestion, sum_p_suggestion,
                                              support_top_suggestion, support_top3_suggestion, sum_support_suggestion,

                                              is_top_suggestion_a_chop, is_top3_suggestion_a_chop, is_top_suggestion_a_complex_procedure, is_top3_suggestion_a_complex_procedure,
                                              p_top3_suggestion_norm, sum_p_suggestion_norm, p_entropy, top3_p_entropy,
                                              ])

            [row.extend([
                sum_p_suggestions_per_case,
                sum_support_suggestions_per_case,
                does_contain_revised_drg,
                n_added_codes,
                n_revised_codes_in_suggestions,
            ]) for row in feature_rows_per_case]
            feature_matrix.extend(feature_rows_per_case)

        df_batch = pd.DataFrame(feature_matrix,
                                columns=['case_id', 'is_case_revised', 'target_drg', 'target_drg_cost_weight', 'delta_target_drg_cost_weight',
                                         'n_suggested_codes', 'n_added_codes_in_suggestions',
                                         'p_top_suggestion', 'p_top3_suggestion', 'sum_p_suggestion',
                                         'support_top_suggestion', 'support_top3_suggestion', 'sum_support_suggestion',
                                         'is_top_suggestion_a_chop', 'is_top3_suggestion_a_chop',
                                         'is_top_suggestion_a_complex_procedure', 'is_top3_suggestion_a_complex_procedure',
                                         'p_top3_suggestion_norm', 'sum_p_suggestion_norm', 'p_entropy', 'top3_p_entropy',
                                         'sum_p_suggestions_per_case', 'sum_support_suggestions_per_case',
                                         'does_contain_revised_drg', 'n_added_codes', 'n_revised_codes_in_suggestions',
                                         ])

        all_data.append(df_batch)

    features = pd.concat(all_data, ignore_index=True) \
        .sort_values(by=['is_case_revised', 'case_id'], ascending=[False, True]) \
        .reset_index(drop=True)

    return features
