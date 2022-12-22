import os.path
import pickle
import shutil
from itertools import chain, combinations
from os.path import join
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

from src.apps.feature_engineering.ccl_sensitivity import calculate_delta_pccl
from src.models.sociodemographics import SOCIODEMOGRAPHIC_ID_COL
from src.service.bfs_cases_db_service import get_all_revised_cases, get_grouped_revisions_for_sociodemographic_ids, \
    get_sociodemographics_by_case_id, get_sociodemographics_by_sociodemographics_ids
from src.service.database import Database

FEATURE_TYPE = np.float32

ONE_HOT_ENCODED_FEATURE_SUFFIX = 'OHE'
RAW_FEATURE_SUFFIX = 'RAW'
RANDOM_SEED = 42

def get_list_of_all_predictors(data: pd.DataFrame, feature_folder: str, *, overwrite: bool = True) -> (dict, dict):
    if overwrite:
        shutil.rmtree(feature_folder, ignore_errors=True)
    Path(feature_folder).mkdir(parents=True, exist_ok=True)

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
    age_encoder = OneHotEncoder(categories=age_bin_label, sparse_output=False, dtype=FEATURE_TYPE, handle_unknown='ignore')
    store_engineered_feature_and_sklearn_encoder('binned_age', age_bin, age_encoder)

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
    delta_ccl_to_next_pccl_feature_filename = __make_feature_filename(decamelize('delta_ccl_to_next_pccl'), RAW_FEATURE_SUFFIX)
    if overwrite or not os.path.exists(delta_ccl_to_next_pccl_feature_filename):
        data = calculate_delta_pccl(data, delta_value_for_max=10.0)
        store_raw_feature('delta_ccl_to_next_pccl')

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


@beartype
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

        revised_cases = sociodemographics_revised_cases[['case_id', 'age_years', 'gender', 'duration_of_stay']].copy()
        revised_cases['revised'] = 1
        logger.info(f'There are {revised_cases.shape[0]} revised cases in the DB')

        revised_cases['case_id'] = revised_cases['case_id'].str.lstrip('0')
        all_data['id'] = all_data['id'].str.lstrip('0')

        revised_cases_in_data = pd.merge(
                revised_cases,
                all_data[['id', 'AnonymerVerbindungskode', 'ageYears', 'gender', 'durationOfStay', 'hospital', 'dischargeYear']].copy(),
                how='outer',
                left_on=('case_id', 'age_years', 'gender', 'duration_of_stay'),
                right_on=('id', 'ageYears', 'gender', 'durationOfStay'),
        )

        # Discard the cases which were revised (according to the DB), but are not present in the data we loaded
        revised_cases_in_data = revised_cases_in_data[~revised_cases_in_data['id'].isna()].reset_index(drop=True)
        # Create the "revised" label column, for modeling
        revised_cases_in_data['is_revised'] = (~revised_cases_in_data['revised'].isna()).astype(int)
        revised_cases_in_data = revised_cases_in_data[['id', 'hospital', 'dischargeYear', 'is_revised']]

        num_revised_cases_in_data = int(revised_cases_in_data["is_revised"].sum())
        num_cases = revised_cases_in_data.shape[0]
        logger.info(f'{num_revised_cases_in_data}/{num_cases} ({float(num_revised_cases_in_data) / num_cases * 100:.1f}%) cases were revised')

        revised_cases_in_data.to_csv(revised_case_ids_filename, index=False)

    else:
        revised_cases_in_data = pd.read_csv(revised_case_ids_filename)

    return revised_cases_in_data


def prepare_train_eval_test_split(dir_output, revised_cases_in_data, hospital_leave_out='KSW', year_leave_out=2020):
    assert hospital_leave_out in revised_cases_in_data['hospital'].values
    assert year_leave_out in revised_cases_in_data['dischargeYear'].values

    # get indices to leave out from training routine for performance app
    y = revised_cases_in_data['is_revised'].values
    ind_hospital_leave_out = np.where((revised_cases_in_data['hospital'].values == hospital_leave_out) &
                                      (revised_cases_in_data['dischargeYear'].values == year_leave_out))[0]
    y_hospital_leave_out = y[ind_hospital_leave_out]

    n_samples = y.shape[0]
    ind_train_test = list(set(range(n_samples)) - set(ind_hospital_leave_out))
    ind_X_train, ind_X_test = train_test_split(ind_train_test, stratify=y[ind_train_test], test_size=0.3,random_state=RANDOM_SEED)
    y_train = y[ind_X_train]
    y_test = y[ind_X_test]

    n_positive_labels_train = int(y_train.sum())

    return ind_X_train, ind_X_test, y_train, y_test, ind_hospital_leave_out, y_hospital_leave_out


def create_performance_app_ground_truth(dir_output: str, revised_cases_in_data: DataFrame, hospital: str, year: int):
    """
    Create performance measing ground truth file for case ranking purposes only.
    @param dir_output: The output directory to store the file in.
    @param revised_cases_in_data: The revised case DataFrame.
    @param hospital: The hospital.
    @param year: The year.
    """
    case_ids_revised = revised_cases_in_data[(revised_cases_in_data['is_revised'] == 1) &
                                             (revised_cases_in_data['hospital'] == hospital) &
                                             (revised_cases_in_data['dischargeYear'] == year)]
    with Database() as db:
        sociodemographic_revised = get_sociodemographics_by_case_id(case_ids_revised['id'].astype(str).values.tolist(), db.session)
        sociodemographic_ids_revised = sociodemographic_revised[SOCIODEMOGRAPHIC_ID_COL]
        grouped_revisions = get_grouped_revisions_for_sociodemographic_ids(sociodemographic_ids_revised, db.session)
        socio_revised_merged = pd.merge(sociodemographic_revised, grouped_revisions, on=SOCIODEMOGRAPHIC_ID_COL, how='right')

    drg_old = list()
    drg_new = list()
    cw_old = list()
    cw_new = list()
    pccl_old = list()
    pccl_new = list()
    for i, row in enumerate(socio_revised_merged.itertuples()):
        if True in row.reviewed and False in row.reviewed:
            ind_original = np.where(np.asarray(row.reviewed) == False)[0][0]
            ind_reviewed = np.where(np.asarray(row.reviewed) == True)[0][0]

            drg_old.append(row.drg[ind_original])
            drg_new.append(row.drg[ind_reviewed])

            cw_old.append(row.effective_cost_weight[ind_original])
            cw_new.append(row.effective_cost_weight[ind_reviewed])

            pccl_old.append(row.pccl[ind_original])
            pccl_new.append(row.pccl[ind_reviewed])
        else:
            drg_old.append('')
            drg_new.append('')
            cw_old.append(0)
            cw_new.append(0)
            pccl_old.append(0)
            pccl_new.append(0)


    n_cases = socio_revised_merged.shape[0]
    placeholder = np.repeat('', n_cases)
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
    ground_truth[ground_truth['cw_delta'] > 0].drop(columns=['cw_delta']).to_csv(join(dir_output, f'ground_truth_performance_app_case_ranking_{hospital}_{year}.csv'), index=False)


def create_predictions_output_performance_app(filename: str, case_ids: ArrayLike, predictions: ArrayLike):
    """
    Write performance measuring output for model to file.
    @param filename: The filename where to store the results.
    @param case_ids: A list of case IDs.
    @param predictions: The probabilities to rank the case IDs.
    """
    pd.DataFrame({
        'CaseId': case_ids,
        'SuggestedCodeRankings': ['']*len(case_ids),
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
