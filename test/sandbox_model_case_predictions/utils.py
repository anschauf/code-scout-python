import os.path
import pickle
from itertools import chain, combinations
from os import listdir
from os.path import join, exists
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from beartype import beartype
# noinspection PyPackageRequirements
from humps import decamelize
from loguru import logger
# noinspection PyProtectedMember
from numpy._typing import ArrayLike
from pandas import DataFrame
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, OrdinalEncoder
from tqdm import tqdm

from src.apps.feature_engineering.ccl_sensitivity import calculate_delta_pccl
from src.models.sociodemographics import SOCIODEMOGRAPHIC_ID_COL
from src.service.bfs_cases_db_service import get_all_reviewed_cases, get_all_revised_cases, \
    get_grouped_revisions_for_sociodemographic_ids, get_sociodemographics_by_case_id, \
    get_sociodemographics_by_sociodemographics_ids
from src.service.database import Database

tqdm.pandas()

FEATURE_TYPE = np.float32

S3_PREFIX = 's3://'
ONE_HOT_ENCODED_FEATURE_SUFFIX = 'OHE'
RAW_FEATURE_SUFFIX = 'RAW'
RANDOM_SEED = 42


# noinspection PyShadowingNames
def get_list_of_all_predictors(
        data: pd.DataFrame,
        feature_folder: str,
        *,
        overwrite: bool = False
        ) -> (dict, dict):

    Path(feature_folder).mkdir(parents=True, exist_ok=True)

    # Store a memory-mapped file for each feature
    features_filenames = dict()
    encoders = dict()

    data = data.sort_values('index', ascending=True)

    # -------------------------------------------------------------------------
    # HELPER FUNCTIONS
    # -------------------------------------------------------------------------
    def __make_feature_filename(standard_name: str, suffix: str) -> str:
        feature_filename = os.path.join(feature_folder, f'{standard_name}_{suffix}.npy')
        features_filenames[f'{standard_name}_{suffix}'] = feature_filename
        return feature_filename

    @beartype
    def store_engineered_feature(standard_name: str,
                                 feature: np.ndarray,
                                 feature_filename: Optional[str] = None,
                                 ):
        if feature_filename is None:
            feature_filename = __make_feature_filename(standard_name, RAW_FEATURE_SUFFIX)

        if overwrite or not os.path.exists(feature_filename):
            if feature.ndim == 1:
                feature = feature.reshape(-1, 1)

            np.save(feature_filename, feature.astype(FEATURE_TYPE), allow_pickle=False, fix_imports=False)
            logger.info(f"Stored the feature '{standard_name}'")
        else:
            logger.debug(f"Ignored existing feature '{standard_name}'")

    @beartype
    def store_engineered_feature_and_sklearn_encoder(standard_name: str,
                                                     feature: np.ndarray,
                                                     encoder,
                                                     feature_filename: Optional[str] = None
                                                     ):
        if feature_filename is None:
            feature_filename = __make_feature_filename(standard_name, RAW_FEATURE_SUFFIX)

        encoder_filename = f'{os.path.splitext(feature_filename)[0]}_encoder.pkl'

        if overwrite or not os.path.exists(feature_filename) or not os.path.exists(encoder_filename):
            if feature.ndim == 1:
                feature = feature.reshape(-1, 1)

            if overwrite or not os.path.exists(feature_filename):
                np.save(feature_filename, feature.astype(FEATURE_TYPE), allow_pickle=False, fix_imports=False)

            if overwrite or not os.path.exists(encoder_filename):
                with open(encoder_filename, 'wb') as f:
                    pickle.dump(encoder, f)

            logger.info(f"Stored the feature '{standard_name}' and its corresponding '{encoder.__class__.__name__}'")

        else:
            with open(encoder_filename, 'rb') as f:
                encoders[__get_full_feature_name(feature_filename)] = pickle.load(f)

            logger.debug(f"Ignored existing feature '{standard_name}'")

    @beartype
    def store_raw_feature(column_name: str):
        if column_name not in data.columns:
            logger.error(f"The column '{column_name}' could not be found in the data")
            return

        standard_name = decamelize(column_name)
        feature_filename = __make_feature_filename(standard_name, RAW_FEATURE_SUFFIX)

        if is_numeric_dtype(data[column_name]):
            store_engineered_feature(standard_name, data[column_name].values, feature_filename)

        else:
            encoder = OrdinalEncoder(dtype=FEATURE_TYPE, handle_unknown='use_encoded_value', unknown_value=-2, encoded_missing_value=-1)
            encoders[__get_full_feature_name(feature_filename)] = encoder
            feature_values = encoder.fit_transform(data[column_name].values.reshape(-1, 1)).ravel()

            store_engineered_feature_and_sklearn_encoder(standard_name, feature_values, encoder, feature_filename)

    @beartype
    def one_hot_encode(column_name: str, *, is_data_uniform: bool = True):
        if column_name not in data.columns:
            logger.error(f"The column '{column_name}' could not be found in the data")
            return

        standard_name = decamelize(column_name)
        feature_filename = __make_feature_filename(standard_name, ONE_HOT_ENCODED_FEATURE_SUFFIX)

        if overwrite or not os.path.exists(feature_filename):
            if is_data_uniform:
                encoder = OneHotEncoder(sparse_output=False, dtype=FEATURE_TYPE, handle_unknown='ignore')
                feature_values = encoder.fit_transform(data[column_name].values.reshape(-1, 1))

            else:
                encoder = MultiLabelBinarizer(sparse_output=False)
                feature_values = encoder.fit_transform(data[column_name].values.tolist())

            encoders[__get_full_feature_name(feature_filename)] = encoder
            store_engineered_feature_and_sklearn_encoder(standard_name, feature_values, encoder, feature_filename)

        else:
            encoder_filename = f'{os.path.splitext(feature_filename)[0]}_encoder.pkl'
            with open(encoder_filename, 'rb') as f:
                encoders[__get_full_feature_name(feature_filename)] = pickle.load(f)

            logger.debug(f"Ignoring existing feature '{standard_name}'")

    # -------------------------------------------------------------------------
    # FEATURE ENGINEERING
    # -------------------------------------------------------------------------
    # TODO Ventilation hour bins: left out, because not much data for ventilation hours
    # TODO difficult to compare medication rate with different units
    # TODO not sure how to get the medication frequency
    # TODO Store number of DRG-relevant codes

    # The following features are stored as-is
    for col_name in (
        # Sociodemographics
        'ageYears', 'ErfassungDerAufwandpunkteFuerIMC', 'hoursMechanicalVentilation', 'AufenthaltIntensivstation',
        'NEMSTotalAllerSchichten', 'durationOfStay', 'leaveDays', 'hospital',
        # DRG-related
        'drgCostWeight', 'effectiveCostWeight', 'NumDrgRelevantDiagnoses', 'NumDrgRelevantProcedures', 'rawPccl',
        'supplementCharges'
        ):
        store_raw_feature(col_name)

    # The following features are stored as raw values as well as one-hot encoded
    for col_name in ('pccl', 'gender'):
        store_raw_feature(col_name)
        one_hot_encode(col_name)

    # The following features are only one-hot encoded
    for col_name in ('Hauptkostenstelle', 'mdc', 'mdcPartition', 'durationOfStayCaseType',
                     'AufenthaltNachAustritt', 'AufenthaltsKlasse', 'Eintrittsart', 'EntscheidFuerAustritt',
                     'AufenthaltsortVorDemEintritt', 'BehandlungNachAustritt', 'EinweisendeInstanz',
                     'HauptkostentraegerFuerGrundversicherungsleistungen',
                     'grouperDischargeCode', 'grouperAdmissionCode'):
        one_hot_encode(col_name)

    # The following features are booleans
    for col_name in ('AufenthaltIntensivstation', 'NEMSTotalAllerSchichten', 'ErfassungDerAufwandpunkteFuerIMC'):
        if col_name not in data.columns:
            logger.error(f"The column '{col_name}' could not be found in the data")
            continue

        has_value = data[col_name].progress_apply(__int_to_bool).values
        store_engineered_feature(decamelize(col_name), has_value)

    if 'IsCaseBelowPcclSplit' not in data.columns:
        logger.error("The column 'IsCaseBelowPcclSplit' could not be found in the data")
    else:
        store_engineered_feature(decamelize('IsCaseBelowPcclSplit'), data['IsCaseBelowPcclSplit'].values)

    # Get the "used" status from the flag columns
    for col_name in ('ageFlag', 'genderFlag', 'durationOfStayFlag', 'grouperAdmissionCodeFlag', 'grouperDischargeCodeFlag', 'hoursMechanicalVentilationFlag', 'gestationAgeFlag'):
        if col_name not in data.columns:
            logger.error(f"The column '{col_name}' could not be found in the data")
            continue

        is_used = data[col_name].str.contains('used').values.astype(FEATURE_TYPE)
        store_engineered_feature(decamelize(col_name), is_used)

    if 'admissionWeightFlag' not in data.columns:
        logger.error("The column 'admissionWeightFlag' could not be found in the data")
    else:
        admission_weight_flag = (~(data['admissionWeightFlag'].str.endswith('_not_used'))).values.astype(FEATURE_TYPE)
        store_engineered_feature(decamelize('admissionWeightFlag'), admission_weight_flag)

    # Other features
    if 'Eintrittsart' not in data.columns:
        logger.error("The column 'Eintrittsart' could not be found in the data")
    else:
        is_emergency_case = (data['Eintrittsart'] == '1').values.astype(FEATURE_TYPE)
        store_engineered_feature('is_emergency_case', is_emergency_case)

    if 'effectiveCostWeight' not in data.columns or 'drgCostWeight' not in data.columns:
        logger.error("The columns 'effectiveCostWeight' and 'drgCostWeight' could not be found in the data")
    else:
        delta_effective_to_base_drg_cost_weight = data['effectiveCostWeight'] - data['drgCostWeight']
        store_engineered_feature('delta_effective_to_base_drg_cost_weight', delta_effective_to_base_drg_cost_weight.values.astype(FEATURE_TYPE))

    if 'procedures' not in data.columns:
        logger.error("The column 'procedures' could not be found in the data")
    else:
        data['number_of_chops'] = data['procedures'].progress_apply(lambda x: len(x))
        store_raw_feature('number_of_chops')

    if 'secondaryDiagnoses' not in data.columns:
        logger.error("The column 'secondaryDiagnoses' could not be found in the data")
    else:
        data['number_of_diags'] = data['secondaryDiagnoses'].progress_apply(lambda x: len(x) + 1)
        store_raw_feature('number_of_diags')

    if 'diagnosesExtendedInfo' not in data.columns:
        logger.error("The column 'diagnosesExtendedInfo' could not be found in the data")
    else:
        data['number_of_diags_ccl_greater_0'] = data['diagnosesExtendedInfo'].progress_apply(__extract_number_of_ccl_greater_null)
        store_raw_feature('number_of_diags_ccl_greater_0')

    if 'proceduresExtendedInfo' not in data.columns:
        logger.error("The column 'proceduresExtendedInfo' could not be found in the data")
    else:
        def _has_complex_procedure(row):
            codes_info = row['proceduresExtendedInfo']
            all_global_functions = set()
            for code_info in codes_info.values():
                all_global_functions.update(code_info['global_functions'])

            complex_procedures = [name for name in all_global_functions if 'kompl' in name.lower()]
            return len(complex_procedures) > 0

        has_complex_procedure = data.progress_apply(_has_complex_procedure, axis=1).values
        store_engineered_feature('has_complex_procedure', has_complex_procedure)

    # Ventilation hours and interactions with it
    if 'hoursMechanicalVentilation' not in data.columns:
        logger.error("The column 'hoursMechanicalVentilation' could not be found in the data")
    else:
        has_ventilation_hours = data['hoursMechanicalVentilation'].progress_apply(__int_to_bool).values.astype(FEATURE_TYPE)
        store_engineered_feature('has_ventilation_hours', has_ventilation_hours)

    if 'mdc' not in data.columns or 'hoursMechanicalVentilation' not in data.columns:
        logger.error("The columns 'mdc' or 'hoursMechanicalVentilation' could not be found in the data")
    else:
        is_drg_in_pre_mdc = (data['mdc'].str.lower() == 'prae').values.astype(FEATURE_TYPE)
        # noinspection PyUnboundLocalVariable
        store_engineered_feature('has_ventilation_hours_AND_in_pre_mdc', has_ventilation_hours * is_drg_in_pre_mdc)

    # Medication information
    # data['medications_atc'] = data['medications'].apply(lambda all_meds: tuple([x.split(':')[0] for x in all_meds]))
    # REF: https://www.wido.de/publikationen-produkte/arzneimittel-klassifikation/
    if 'medications' not in data.columns:
        logger.error("The column 'medications' could not be found in the data")
    else:
        data['medications_atc3'] = data['medications'].progress_apply(lambda all_meds: tuple([x.split(':')[0][:3] for x in all_meds]))
        data['medications_kind'] = data['medications'].progress_apply(lambda all_meds: tuple([x.split(':')[2] for x in all_meds]))
        # one_hot_encode('medications_atc', is_data_uniform=False)
        one_hot_encode('medications_atc3', is_data_uniform=False)
        one_hot_encode('medications_kind', is_data_uniform=False)

    # age bins of patient
    if 'ageYears' not in data.columns or 'ageDays' not in data.columns:
        logger.error("The columns 'ageYears' and 'ageDays' could not be found in the data")
    else:
        age_bin, age_bin_label = categorize_age(data['ageYears'].values, data['ageDays'].values)
        age_encoder = OneHotEncoder(categories=age_bin_label, sparse_output=False, dtype=FEATURE_TYPE, handle_unknown='ignore')
        store_engineered_feature_and_sklearn_encoder('binned_age', age_bin, age_encoder)

    # Noisy features, which are very unlikely to be correlated but they can be used to evaluate training performance
    if 'entryDate' not in data.columns:
        logger.error("The column 'entryDate' could not be found in the data")
    else:
        data['month_admission'] = data['entryDate'].progress_apply(lambda date: date[4:6])
        one_hot_encode('month_admission')

    if 'exitDate' not in data.columns:
        logger.error("The column 'exitDate' could not be found in the data")
    else:
        data['month_discharge'] = data['exitDate'].progress_apply(lambda date: date[4:6])
        one_hot_encode('month_discharge')

        data['year_discharge'] = data['exitDate'].progress_apply(lambda date: date[:4])
        store_raw_feature('year_discharge')

    # Calculate and store the CCL-sensitivity of the cases
    if 'year_discharge' not in data.columns or 'pccl' not in data.columns or 'rawPccl' not in data.columns:
        logger.error("The columns 'year_discharge', 'pccl' and 'rawPccl' could not be found in the data")
    else:
        delta_ccl_to_next_pccl_feature_filename = __make_feature_filename('delta_ccl_to_next_pccl', RAW_FEATURE_SUFFIX)
        if overwrite or not os.path.exists(delta_ccl_to_next_pccl_feature_filename):
            data = calculate_delta_pccl(data, delta_value_for_max=10.0)
            store_raw_feature('delta_ccl_to_next_pccl')

    if 'VectorizedCodes' not in data.columns:
        logger.error("The columns 'VectorizedCodes' could not be found in the data")
    else:
        vectors_filename = __make_feature_filename('vectorized_codes', ONE_HOT_ENCODED_FEATURE_SUFFIX)
        vectors_encoder_filename = f'{os.path.splitext(vectors_filename)[0]}_encoder.pkl'

        if overwrite or not os.path.exists(vectors_filename) or not os.path.exists(vectors_encoder_filename):
            # List all the indices of the sparse vectors across all vectors (basically ignoring empty columns)
            all_indices = set()
            # noinspection PyShadowingNames
            def get_vector_indices(sparse_vector_info):
                all_indices.update(sparse_vector_info['indices'])
            data['VectorizedCodes'].progress_apply(lambda sparse_vector_info: get_vector_indices(sparse_vector_info))

            vector_size = len(all_indices)
            logger.debug(f'Will create dense vectors with length {vector_size}')
            # Map each sparse index to a position in the compressed dense vector
            all_indices = {idx: pos for pos, idx in enumerate(sorted(list(all_indices)))}

            logger.debug('Storing all the vectors in a memory-mapped location ...')
            temp_vectors_filename = os.path.splitext(vectors_filename)[0] + '_memmap.npy'
            vectors = np.memmap(temp_vectors_filename, dtype=FEATURE_TYPE, mode='w+', shape=(data.shape[0], vector_size))

            for idx, (_, sparse_vector_info) in enumerate(tqdm(data['VectorizedCodes'].items(), total=data.shape[0])):
                vector = np.zeros(vector_size, dtype=FEATURE_TYPE)
                vector_indices = [all_indices[real_idx] for real_idx in sparse_vector_info['indices']]
                vector[vector_indices] = sparse_vector_info['values']

                vectors[idx, :] = vector

            logger.debug('Moving all the vectors to a pickled file ...')
            # Files stored with `np.save()` contain enough information to be reloaded without knowing their shape or
            # data-type
            np.save(vectors_filename, vectors, allow_pickle=False, fix_imports=False)

            logger.debug('Deleting the temporary file ...')
            del vectors
            os.remove(temp_vectors_filename)

            vectors_encoder = OneHotEncoder(categories=sorted(list(all_indices.keys())), sparse_output=False, dtype=FEATURE_TYPE, handle_unknown='ignore')
            if overwrite or not os.path.exists(vectors_encoder_filename):
                with open(vectors_encoder_filename, 'wb') as f:
                    pickle.dump(vectors_encoder, f)

            logger.info(f"Stored the feature 'vectorized_codes' and its corresponding '{vectors_encoder.__class__.__name__}'")

        else:
            with open(vectors_encoder_filename, 'rb') as f:
                encoders[__get_full_feature_name(vectors_filename)] = pickle.load(f)

            logger.debug(f"Ignored existing feature 'vectorized_codes'")

    return features_filenames, encoders


def __int_to_bool(row):
    if isinstance(row, int):
        return row > 0
    else:
        return False


def __extract_number_of_ccl_greater_null(row: dict):
    num_ccl_greater_null = 0
    for key in row.keys():
        if row[key]['ccl'] > 0:
            num_ccl_greater_null += 1
    return num_ccl_greater_null


def categorize_age(ages_years: npt.ArrayLike, ages_days: npt.ArrayLike):
    """ Compute age bins

    @param ages_years: The age in years.
    @param ages_days: The age in days.
    :return: (The age-categorized BFS data, the labels for the age bins)
    """
    agebins_labels = ['age_below_28_days', 'age_28_days_to_2_years', 'age_2_to_5_years', 'age_6_to_15_years',
                      'age_16_to_29_years', 'age_30_to_39_years', 'age_40_to_49_years', 'age_50_to_59_years',
                      'age_60_to_69_years', 'age_70_to_79_years', 'age_80_and_older']
    categories_age = np.zeros((len(ages_years), len(agebins_labels)))
    for i, age_year_days in enumerate(zip(ages_years, ages_days)):
        age_year = age_year_days[0]
        age_day = age_year_days[1]

        if age_year == 0 and 0 <= age_day < 28:
            categories_age[i, 0] = 1
        elif 28 <= age_day <= 365 or 1 <= age_year < 2:
            categories_age[i, 1] = 1
        elif 2 <= age_year <= 5:
            categories_age[i, 2] = 1
        elif 6 <= age_year <= 15:
            categories_age[i, 3] = 1
        elif 16 <= age_year <= 29:
            categories_age[i, 4] = 1
        elif 30 <= age_year <= 39:
            categories_age[i, 5] = 1
        elif 40 <= age_year <= 49:
            categories_age[i, 6] = 1
        elif 50 <= age_year <= 59:
            categories_age[i, 7] = 1
        elif 60 <= age_year <= 69:
            categories_age[i, 8] = 1
        elif 70 <= age_year <= 79:
            categories_age[i, 9] = 1
        else:
            categories_age[i, 10] = 1

    return categories_age, agebins_labels


def list_all_subsets(ss, *, reverse: bool = False):
    if reverse:
        index_range = range(len(ss) + 1, 0, -1)
    else:
        index_range = range(0, len(ss) + 1)

    return chain(*map(lambda x: combinations(ss, x), index_range))


def __get_full_feature_name(feature_filename: str) -> str:
    return os.path.splitext(os.path.basename(feature_filename))[0]


def get_revised_case_ids(all_data: pd.DataFrame,
                         revised_case_ids_filename: str,
                         *,
                         overwrite: bool = False
                         ) -> pd.DataFrame:

    if overwrite or not os.path.exists(revised_case_ids_filename):
        with Database() as db:
            revised_cases_all = get_all_revised_cases(db.session)
            revised_case_sociodemographic_ids = revised_cases_all['sociodemographic_id'].values.tolist()
            sociodemographics_revised_cases = get_sociodemographics_by_sociodemographics_ids(revised_case_sociodemographic_ids, db.session)

            reviewed_cases_all = get_all_reviewed_cases(db.session)
            reviewed_case_sociodemographic_ids = reviewed_cases_all['sociodemographic_id'].values.tolist()
            sociodemographics_reviewed_cases = get_sociodemographics_by_sociodemographics_ids(reviewed_case_sociodemographic_ids, db.session)

        revised_cases = sociodemographics_revised_cases[['case_id', 'age_years', 'gender', 'duration_of_stay']].copy()
        revised_cases['revised'] = 1
        logger.info(f'There are {revised_cases.shape[0]} revised cases in the DB')

        reviewed_cases = sociodemographics_reviewed_cases[['case_id', 'age_years', 'gender', 'duration_of_stay']].copy()
        reviewed_cases['reviewed'] = 1
        logger.info(f'There are {reviewed_cases.shape[0]} reviewed cases in the DB')

        revised_cases['case_id'] = revised_cases['case_id'].str.lstrip('0')
        reviewed_cases['case_id'] = reviewed_cases['case_id'].str.lstrip('0')
        all_data['id'] = all_data['id'].str.lstrip('0')

        revised_cases_in_data = pd.merge(
            revised_cases,
            all_data[['id', 'AnonymerVerbindungskode', 'ageYears', 'gender', 'durationOfStay', 'hospital', 'dischargeYear', 'index']].copy(),
            how='outer',
            left_on=('case_id', 'age_years', 'gender', 'duration_of_stay'),
            right_on=('id', 'ageYears', 'gender', 'durationOfStay'),
        )

        revised_cases_in_data = pd.merge(
            revised_cases_in_data,
            reviewed_cases,
            how='outer',
            left_on=('id', 'ageYears', 'gender', 'durationOfStay'),
            right_on=('case_id', 'age_years', 'gender', 'duration_of_stay'),
        )

        revised_cases_in_data['revised'].fillna(0, inplace=True)
        revised_cases_in_data['reviewed'].fillna(0, inplace=True)

        # Discard cases which did not appear in the loaded data
        revised_cases_in_data = revised_cases_in_data[~revised_cases_in_data['index'].isna()]

        # Discard the cases which were revised or reviewed (according to the DB), but are not present in the data we loaded
        revised_cases_in_data = revised_cases_in_data[~revised_cases_in_data['id'].isna()]
        # Create the label columns
        revised_cases_in_data['is_revised'] = revised_cases_in_data['revised'].astype(int)
        revised_cases_in_data['is_reviewed'] = revised_cases_in_data['reviewed'].astype(int)
        revised_cases_in_data = revised_cases_in_data[['index', 'id', 'hospital', 'dischargeYear', 'is_revised', 'is_reviewed']]

        num_revised_cases_in_data = int(revised_cases_in_data['is_revised'].sum())
        num_reviewed_cases_in_data = int(revised_cases_in_data['is_reviewed'].sum())
        num_cases = revised_cases_in_data.shape[0]
        logger.info(f'{num_revised_cases_in_data}/{num_cases} ({float(num_revised_cases_in_data) / num_cases * 100:.1f}%) cases were revised')
        logger.info(f'{num_reviewed_cases_in_data}/{num_cases} ({float(num_reviewed_cases_in_data) / num_cases * 100:.1f}%) cases were reviewed')

        # Re-sort the joined dataset according to the original order
        revised_cases_in_data = revised_cases_in_data.sort_values('index', ascending=True).reset_index(drop=True)

        revised_cases_in_data.to_csv(revised_case_ids_filename, index=False)

    else:
        revised_cases_in_data = pd.read_csv(revised_case_ids_filename, dtype='string[pyarrow]')

    # Set the correct data types
    revised_cases_in_data['index'] = revised_cases_in_data['index'].astype(int)
    revised_cases_in_data['id'] = revised_cases_in_data['id'].astype(str)
    revised_cases_in_data['hospital'] = revised_cases_in_data['hospital'].astype(str)
    revised_cases_in_data['dischargeYear'] = revised_cases_in_data['dischargeYear'].astype(int)
    revised_cases_in_data['is_revised'] = revised_cases_in_data['is_revised'].astype(int)
    revised_cases_in_data['is_reviewed'] = revised_cases_in_data['is_reviewed'].astype(int)

    # Sort by index to keep the alignment correct with the feature-files
    revised_cases_in_data.sort_values('index', ascending=True, inplace=True)

    return revised_cases_in_data


def prepare_train_eval_test_split(dir_output, revised_cases_in_data, hospital_leave_out='KSW', year_leave_out=2020, only_reviewed_cases=False):
    assert hospital_leave_out in revised_cases_in_data['hospital'].values
    assert year_leave_out in revised_cases_in_data['dischargeYear'].values
    revised_cases_in_data['id'] = revised_cases_in_data['id'].astype('string')

    # get indices to leave out from training routine for performance app
    y = revised_cases_in_data['is_revised'].values
    ind_hospital_leave_out = np.where((revised_cases_in_data['hospital'].values == hospital_leave_out) &
                                      (revised_cases_in_data['dischargeYear'].values == year_leave_out))[0]
    y_hospital_leave_out = y[ind_hospital_leave_out]

    n_samples = y.shape[0]
    if only_reviewed_cases:
        ind_not_validated = list(np.where(np.logical_and(revised_cases_in_data['is_reviewed'] == 0, revised_cases_in_data['is_revised'] == 0))[0])
        ind_train_test = list(set(range(n_samples)) - set(ind_hospital_leave_out) - set(ind_not_validated))
    else:
        ind_train_test = list(set(range(n_samples)) - set(ind_hospital_leave_out))
    ind_X_train, ind_X_test = train_test_split(ind_train_test, stratify=y[ind_train_test], test_size=0.3,random_state=RANDOM_SEED)
    y_train = y[ind_X_train]
    y_test = y[ind_X_test]

    return ind_X_train, ind_X_test, y_train, y_test, ind_hospital_leave_out, y_hospital_leave_out


def prepare_train_eval_test_split_p(revised_cases_in_data: pd.DataFrame,
                                   hospital_leave_out: Optional[str],
                                   year_leave_out: Optional[int],
                                   *,
                                   only_reviewed_cases: bool = False
                                   ):
    if hospital_leave_out is not None and hospital_leave_out not in revised_cases_in_data['hospital'].values:
        raise ValueError(f'Unknown hospital {hospital_leave_out}')

    if year_leave_out is not None and year_leave_out not in revised_cases_in_data['dischargeYear'].values:
        raise ValueError(f'Unknown year {year_leave_out}')

    y = revised_cases_in_data['is_revised'].values

    ind_hospital_leave_out_filter = set(np.arange(revised_cases_in_data.shape[0]))
    if hospital_leave_out is not None:
        ind_hospital_leave_out_filter.intersection_update(
            np.where(revised_cases_in_data['hospital'] == hospital_leave_out)[0])
    if year_leave_out is not None:
        ind_hospital_leave_out_filter.intersection_update(
            np.where(revised_cases_in_data['dischargeYear'] == year_leave_out)[0])
    ind_hospital_leave_out = np.sort(list(ind_hospital_leave_out_filter))
    y_hospital_leave_out = y[ind_hospital_leave_out]

    # get case_ids for reviewed but not revised cases
    if only_reviewed_cases:
        all_true_ground_truth_data = revised_cases_in_data[
            (revised_cases_in_data['is_revised'] == 1) | (revised_cases_in_data['is_reviewed'] == 1)]
        ind_train_test = all_true_ground_truth_data['index'].values

    else:
        n_samples = y.shape[0]
        ind_train_test = np.setdiff1d(np.arange(n_samples), ind_hospital_leave_out)

    ind_train, ind_eval = train_test_split(ind_train_test, stratify=y[ind_train_test], test_size=0.3,
                                           random_state=RANDOM_SEED)

    y_train = y[ind_train]
    y_eval = y[ind_eval]

    return ind_train, ind_eval, y_train, y_eval, ind_hospital_leave_out, y_hospital_leave_out


# noinspection PyUnresolvedReferences
def create_performance_app_ground_truth(dir_output: str, revised_cases_in_data: DataFrame, hospital: Optional[str], year: Optional[int], overwrite=False):
    """
    Create performance ground truth file for case ranking purposes only.
    @param dir_output: The output directory to store the file in.
    @param revised_cases_in_data: The revised case DataFrame.
    @param hospital: The hospital.
    @param year: The year.
    """
    case_ids_revised_filter = revised_cases_in_data['is_revised'] == 1

    if hospital is not None:
        case_ids_revised_filter &= revised_cases_in_data['hospital'] == hospital
        hospital_msg = hospital
    else:
        hospital_msg = 'all_hospitals'

    if year is not None:
        case_ids_revised_filter &= revised_cases_in_data['dischargeYear'] == year
        year_msg = year
    else:
        year_msg = 'all_years'

    filename = join(dir_output, f'ground_truth_performance_app_case_ranking_{hospital_msg}_{year_msg}.csv')
    if not np.logical_and(exists(filename) == True, overwrite == False):
        case_ids_revised = revised_cases_in_data[case_ids_revised_filter]

        logger.info(f'Selected {case_ids_revised.shape[0]} cases from {hospital_msg} ({year_msg})')

        with Database() as db:
            logger.trace('Reading the sociodemographics from the DB ...')
            sociodemographic_revised = get_sociodemographics_by_case_id(case_ids_revised['id'].values.tolist(), db.session)
            sociodemographic_ids_revised = sociodemographic_revised[SOCIODEMOGRAPHIC_ID_COL]

            logger.trace('Reading and grouping the revisions ...')
            grouped_revisions = get_grouped_revisions_for_sociodemographic_ids(sociodemographic_ids_revised, db.session)
            socio_revised_merged = pd.merge(sociodemographic_revised, grouped_revisions, on=SOCIODEMOGRAPHIC_ID_COL)

        logger.info('Assembling info on revisions ...')
        n_rows = socio_revised_merged.shape[0]

        drg_old = np.empty(n_rows, dtype=object)
        drg_new = np.empty_like(drg_old)

        cw_old = np.empty(n_rows, dtype=float)
        cw_new = np.empty_like(cw_old)

        pccl_old = np.empty(n_rows, dtype=int)
        pccl_new = np.empty_like(pccl_old)

        for i, row in enumerate(tqdm(socio_revised_merged.itertuples())):
            is_revised = set(row.reviewed) == {True, False}

            if is_revised:
                ind_original = row.reviewed.index(False)
                ind_reviewed = row.reviewed.index(True)

                drg_old[i] = row.drg[ind_original]
                drg_new[i] = row.drg[ind_reviewed]

                cw_old[i] = row.effective_cost_weight[ind_original]
                cw_new[i] = row.effective_cost_weight[ind_reviewed]

                pccl_old[i] = row.pccl[ind_original]
                pccl_new[i] = row.pccl[ind_reviewed]

            else:
                drg_old[i] = ''
                drg_new[i] = ''
                cw_old[i] = 0
                cw_new[i] = 0
                pccl_old[i] = 0
                pccl_new[i] = 0

        placeholder = np.repeat('', n_rows)
        ground_truth = pd.DataFrame({
            'CaseId': socio_revised_merged['case_id'].values,
            'AdmNo': placeholder,
            'FID': placeholder,
            'PatID': placeholder,
            'ICD_added': placeholder,
            'ICD_dropped': placeholder,
            'CHOP_added': placeholder,
            'CHOP_dropped': placeholder,
            'DRG_old': drg_old,
            'DRG_new': drg_new,
            'CW_old': cw_old,
            'CW_new': cw_new,
            'PCCL_old': pccl_old,
            'PCCL_new': pccl_new
        })

        #TODO maybe remove this filter again, just for the time being till we checked the revised cases in the DB
        ground_truth = ground_truth[ground_truth['CW_new'] > 0]
        ground_truth['cw_delta'] = ground_truth['CW_new'] - ground_truth['CW_old']
        ground_truth[ground_truth['cw_delta'] > 0] \
            .drop(columns=['cw_delta']) \
            .to_csv(filename, index=False)


def create_predictions_output_performance_app(filename: str, case_ids: ArrayLike, predictions: ArrayLike):
    """
    Write performance measuring output for model to file.
    @param filename: The filename where to store the results.
    @param case_ids: A list of case IDs.
    @param predictions: The probabilities to rank the case IDs.
    """
    pd.DataFrame({
        'CaseId': case_ids,
        'SuggestedCodeRankings': [''] * len(case_ids),
        'UpcodingConfidenceScore': predictions
    }).to_csv(filename, index=False, sep=';')


def list_all_feature_names(all_data: pd.DataFrame, features_dir: str, feature_indices: Optional[list] = None) -> list:
    feature_filenames, encoders = get_list_of_all_predictors(all_data, features_dir, overwrite=False)
    feature_names = sorted(list(feature_filenames.keys()))

    all_feature_names = list()

    if feature_indices is None:
        feature_indices = list(range(len(feature_names)))

    for feature_idx in feature_indices:
        feature_name = feature_names[feature_idx]
        feature_name_wo_suffix = '_'.join(feature_name.split('_')[:-1])

        if feature_name in encoders:
            encoder = encoders[feature_name]
            if isinstance(encoder, MultiLabelBinarizer):
                encoded_names = encoder.classes_

            else:
                if isinstance(encoder.categories, str) and encoder.categories == 'auto':
                    encoded_names = encoder.categories_[0]
                else:
                    encoded_names = encoder.categories

            all_feature_names.extend(f'{feature_name_wo_suffix}="{encoded_name}"' for encoded_name in encoded_names)

        else:
            all_feature_names.append(feature_name_wo_suffix)

    return all_feature_names


def get_screen_summary_random_forest(RESULTS_DIR):
    all_runs = listdir(RESULTS_DIR)
    all_runs = [f for f in all_runs if f.startswith('n_trees')]

    n_estimator = list()
    max_depth = list()
    min_sample_leaf = list()
    min_sample_split = list()
    precision_train = list()
    precision_test = list()
    recall_train = list()
    recall_test = list()
    f1_train = list()
    f1_test = list()

    for file in all_runs:
        full_filename = join(RESULTS_DIR, file, 'performance.txt')

        if exists(full_filename):
            split_name = file.split('-')
            n_estimator.append(int(split_name[0].split('_')[-1]))
            max_depth.append(int(split_name[1].split('_')[-1]))
            min_sample_leaf.append(int(split_name[2].split('_')[-1]))
            if len(split_name) > 3:
                min_sample_split.append(int(split_name[3].split('_')[-1]))
            else:
                min_sample_split.append('')

            with open(full_filename) as f:
                lines = f.readlines()

            precision_train.append(float(lines[4].split(',')[0].split(' ')[-1]))
            precision_test.append(float(lines[4].split(',')[1].split(' ')[-1]))

            recall_train.append(float(lines[5].split(',')[0].split(' ')[-1]))
            recall_test.append(float(lines[5].split(',')[1].split(' ')[-1]))

            f1_train.append(float(lines[6].split(',')[0].split(' ')[-1]))
            f1_test.append(float(lines[6].split(',')[1].split(' ')[-1]))

    summary = pd.DataFrame({
        'n_estimator': n_estimator,
        'max_depth': max_depth,
        'min_sample_leaf': min_sample_leaf,
        'min_sample_split': min_sample_split,
        'precision_train': precision_train,
        'precision_test': precision_test,
        'precision_train-test': np.asarray(precision_train) - np.asarray(precision_test),
        'recall_train': recall_train,
        'recall_test': recall_test,
        'recall_train-test': np.asarray(recall_train) - np.asarray(recall_test),
        'f1_train': f1_train,
        'f1_test': f1_test,
        'f1_train-test': np.asarray(f1_train) - np.asarray(f1_test)
    }).sort_values('f1_test', ascending=False)
    summary.to_csv(join(RESULTS_DIR, 'screen_summary.csv'), index=False)

    return summary
