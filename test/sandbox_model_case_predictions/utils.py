import os.path
import pickle
from itertools import chain, combinations
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from beartype import beartype
from humps import decamelize
from loguru import logger
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, OrdinalEncoder

from src.apps.feature_engineering.ccl_sensitivity import calculate_delta_pccl

FEATURE_TYPE = np.float32

ONE_HOT_ENCODED_FEATURE_SUFFIX = 'OHE'
RAW_FEATURE_SUFFIX = 'RAW'


def get_list_of_all_predictors(data: pd.DataFrame, feature_folder: str, *, overwrite: bool = True) -> (dict, dict):
    # Store a memory-mapped file for each feature
    features_filenames = dict()
    encoders = dict()

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

            np.save(feature_filename, feature.astype(FEATURE_TYPE), allow_pickle=False, fix_imports=False)

            with open(encoder_filename, 'wb') as f:
                pickle.dump(encoder, f)

            logger.info(f"Stored the feature '{standard_name}' and its corresponding '{encoder.__class__.__name__}'")

        else:
            with open(encoder_filename, 'rb') as f:
                encoders[__get_full_feature_name(feature_filename)] = pickle.load(f)

            logger.debug(f"Ignored existing feature '{standard_name}'")


    @beartype
    def store_raw_feature(column_name: str):
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
    def one_hot_encode(colum_name: str, *, is_data_uniform: bool = True):
        standard_name = decamelize(colum_name)
        feature_filename = __make_feature_filename(standard_name, ONE_HOT_ENCODED_FEATURE_SUFFIX)

        if overwrite or not os.path.exists(feature_filename):
            if is_data_uniform:
                encoder = OneHotEncoder(sparse_output=False, dtype=FEATURE_TYPE, handle_unknown='ignore')
                feature_values = encoder.fit_transform(data[colum_name].values.reshape(-1, 1))

            else:
                encoder = MultiLabelBinarizer(sparse_output=False)
                feature_values = encoder.fit_transform(data[colum_name].values.tolist())

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
    # The following features are stored as-is
    for col_name in (
        # Sociodemographics
        'ageYears', 'ErfassungDerAufwandpunkteFuerIMC', 'hoursMechanicalVentilation', 'AufenthaltIntensivstation',
        'NEMSTotalAllerSchichten', 'durationOfStay', 'leaveDays',
        # DRG-related
        'drgCostWeight', 'effectiveCostWeight', 'NumDrgRelevantDiagnoses', 'NumDrgRelevantProcedures', 'rawPccl',
        'supplementCharges'
        ):
        store_raw_feature(col_name)

    # The following features are stored as raw values as well as one-hot encoded
    for col_name in ('pccl', ):
        store_raw_feature(col_name)
        one_hot_encode(col_name)

    # The following features are only one-hot encoded
    for col_name in ('gender', 'Hauptkostenstelle', 'mdc', 'mdcPartition', 'durationOfStayCaseType',
                     'AufenthaltNachAustritt', 'AufenthaltsKlasse', 'Eintrittsart', 'EntscheidFuerAustritt',
                     'AufenthaltsortVorDemEintritt', 'BehandlungNachAustritt', 'EinweisendeInstanz',
                     'EntscheidFuerAustritt', 'HauptkostentraegerFuerGrundversicherungsleistungen',
                     'grouperDischargeCode', 'grouperAdmissionCode'):
        one_hot_encode(col_name)

    # The following features are booleans
    for col_name in ('AufenthaltIntensivstation', 'NEMSTotalAllerSchichten', 'ErfassungDerAufwandpunkteFuerIMC'):
        has_value = data[col_name].apply(__int_to_bool).values
        store_engineered_feature(decamelize(col_name), has_value)

    store_engineered_feature(decamelize('IsCaseBelowPcclSplit'), data['IsCaseBelowPcclSplit'].values)

    # Get the "used" status from the flag columns
    for col_name in ('ageFlag', 'genderFlag', 'durationOfStayFlag', 'grouperAdmissionCodeFlag', 'grouperDischargeCodeFlag', 'hoursMechanicalVentilationFlag', 'gestationAgeFlag'):
        is_used = data[col_name].str.contains('used').values.astype(FEATURE_TYPE)
        store_engineered_feature(decamelize(col_name), is_used)

    admission_weight_flag = (~(data['admissionWeightFlag'].str.endswith('_not_used'))).values.astype(FEATURE_TYPE)
    store_engineered_feature(decamelize('admissionWeightFlag'), admission_weight_flag)

    # Other features
    is_emergency_case = (data['Eintrittsart'] == '1').values.astype(FEATURE_TYPE)
    store_engineered_feature('is_emergency_case', is_emergency_case)

    delta_effective_to_base_drg_cost_weight = data['effectiveCostWeight'] - data['drgCostWeight']
    store_engineered_feature('delta_effective_to_base_drg_cost_weight', delta_effective_to_base_drg_cost_weight.values.astype(FEATURE_TYPE))

    data['number_of_chops'] = data['procedures'].apply(lambda x: len(x))
    store_raw_feature('number_of_chops')
    data['number_of_diags'] = data['secondaryDiagnoses'].apply(lambda x: len(x) + 1)
    store_raw_feature('number_of_diags')
    data['number_of_diags_ccl_greater_0'] = data['diagnosesExtendedInfo'].apply(__extract_number_of_ccl_greater_null)
    store_raw_feature('number_of_diags_ccl_greater_0')

    def _has_complex_procedure(row):
        codes_info = row['proceduresExtendedInfo']
        all_global_functions = set()
        for code_info in codes_info.values():
            all_global_functions.update(code_info['global_functions'])

        complex_procedures = [name for name in all_global_functions if 'kompl' in name.lower()]
        return len(complex_procedures) > 0

    has_complex_procedure = data.apply(_has_complex_procedure, axis=1).values
    store_engineered_feature('has_complex_procedure', has_complex_procedure)

    # Ventilation hours and interactions with it
    has_ventilation_hours = data['hoursMechanicalVentilation'].apply(__int_to_bool).values.astype(FEATURE_TYPE)
    store_engineered_feature('has_ventilation_hours', has_ventilation_hours)
    is_drg_in_pre_mdc = (data['mdc'].str.lower() == 'prae').values.astype(FEATURE_TYPE)
    store_engineered_feature('has_ventilation_hours_AND_in_pre_mdc', has_ventilation_hours * is_drg_in_pre_mdc)

    # Medication information
    # data['medications_atc'] = data['medications'].apply(lambda all_meds: tuple([x.split(':')[0] for x in all_meds]))
    # REF: https://www.wido.de/publikationen-produkte/arzneimittel-klassifikation/
    data['medications_atc3'] = data['medications'].apply(lambda all_meds: tuple([x.split(':')[0][:3] for x in all_meds]))
    data['medications_kind'] = data['medications'].apply(lambda all_meds: tuple([x.split(':')[2] for x in all_meds]))
    # one_hot_encode('medications_atc', is_data_uniform=False)
    one_hot_encode('medications_atc3', is_data_uniform=False)
    one_hot_encode('medications_kind', is_data_uniform=False)

    # data['adrg'] = data['drg'].apply(lambda x: x[:3])
    # one_hot_encode('adrg')

    # age bins of patient
    age_bin, age_bin_label = categorize_age(data['ageYears'].values, data['ageDays'].values)
    store_engineered_feature('binned_age', age_bin)

    # TODO Ventilation hour bins: left out, because not much data for ventilation hours
    # TODO difficult to compare medication rate with different units
    # TODO not sure how to get the medication frequency

    # Noisy features, which are very unlikely to be correlated but they can be used to evaluate training performance
    # data['day_admission'] = data['entryDate'].apply(lambda date: datetime(int(date[:4]), int(date[4:6]), int(date[6:])).weekday())
    # data['day_discharge'] = data['exitDate'].apply(lambda date: datetime(int(date[:4]), int(date[4:6]), int(date[6:])).weekday())
    data['month_admission'] = data['entryDate'].apply(lambda date: date[4:6])
    data['month_discharge'] = data['exitDate'].apply(lambda date: date[4:6])
    data['year_discharge'] = data['exitDate'].apply(lambda date: date[:4])
    # one_hot_encode('day_admission')
    # one_hot_encode('day_discharge')
    one_hot_encode('month_admission')
    one_hot_encode('month_discharge')
    # one_hot_encode('year_discharge')

    # Calculate and store the CCL-sensitivity of the cases
    data = calculate_delta_pccl(data, delta_value_for_max=10.0)
    store_raw_feature('delta_ccl_to_next_pccl_col')

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
    agebins_labels = ['age_below_28_days', 'age_28_days_to_2_years', 'age_2_to_5_years', 'age_6_to_15_years', 'age_16_to_29_years', 'age_30_to_39_years', 'age_40_to_49_years', 'age_50_to_59_years', 'age_60_to_69_years', 'age_70_to_79_years', 'age_80_and_older']
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
