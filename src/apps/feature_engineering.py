from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from apps.feature_engineering.medications import get_atc_codes
from revised_case_normalization.notebook_functions.global_configs import VENTILATION_HOURS_COL
from src.apps.feature_engineering.ccl_sensitivity import calculate_delta_pccl
from src.apps.feature_engineering.utils import create_feature_engineering_table, store_features_in_db, validate_app_args
from src.data_model.feature_engineering import AGE_BINNED_COL, EMERGENCY_COL, HAS_HOURS_IN_ICU_COL, \
    HAS_IMC_EFFORT_POINTS_COL, HAS_NEMS_POINTS_COL, HAS_VENTILATION_HOURS_COL, VENTILATION_HOURS_ADRG_NO_A_COL
from src.service.bfs_cases_db_service import read_cases_in_chunks,\
    read_cases_by_sociodemographic_id
from src.service.database import Database


def create_all_features(*, chunksize: int, n_rows: Optional[int] = None):
    chunksize, n_rows = validate_app_args(chunksize, n_rows)
    columns_to_select = create_feature_engineering_table()

    all_features = None

    with Database() as db:
        for cases in read_cases_by_sociodemographic_id(db.session, n_rows=n_rows, chunksize=chunksize, lower=800001, upper=860000):

            # --- Engineer features ---

            # Calculating delta pccl
            cases = calculate_delta_pccl(cases, delta_value_for_max=np.nan)

            # Ventilation hours
            cases[VENTILATION_HOURS_COL] = cases[VENTILATION_HOURS_COL].fillna(0)
            cases[HAS_VENTILATION_HOURS_COL] = cases[VENTILATION_HOURS_COL] > 0

            # 'has_ventilation_hours' (boolean) * aDRG not starting with "A"
            cases['adrg_no_A'] = cases['adrg'].str[0]
            cases['adrg_no_A'] = cases['adrg_no_A'] != "A"
            cases[VENTILATION_HOURS_ADRG_NO_A_COL] = cases['adrg_no_A'] * cases[HAS_VENTILATION_HOURS_COL]

            # Emergency Boolean
            cases[EMERGENCY_COL] = cases['admission_type'] == "1"

            # Hours in ICU Boolean
            cases[HAS_HOURS_IN_ICU_COL] = cases['hours_in_icu'] > 0

            # Has IMC effort points boolean
            cases[HAS_IMC_EFFORT_POINTS_COL] = cases['imc_effort_points'] > 0

            # create age bins
            age_bins = [0, 1, 2, 5, 9, 10, 15, 16, 19, 20, 29, 30, 40, 49, 50, 59, 60, 69, 70, 79, 80, 89, 90, 99, 100, 120]
            cases[AGE_BINNED_COL] = pd.cut(cases['age_years'], age_bins)
            cases[AGE_BINNED_COL] = cases[AGE_BINNED_COL].astype(str).str.replace('nan', '0')

            # Nems Boolean
            cases[HAS_NEMS_POINTS_COL] = cases['nems_total'] > 0

            # Weekday from date admission
            cases['admission_date'] = pd.to_datetime(cases['admission_date'])
            cases['admission_date_weekday'] = cases['admission_date'].dt.day_name()

            # Month of admission date
            cases['admission_date_month'] = cases['admission_date'].dt.month_name()

            # Weekday from discharge date
            cases['discharge_date'] = pd.to_datetime(cases['discharge_date'])
            cases['discharge_date_weekday'] = cases['discharge_date'].dt.day_name()

            # Month of admission date
            cases['discharge_date_month'] = cases['discharge_date'].dt.month_name()

            cases = get_atc_codes(cases)

            # --- Store the data in memory ---
            if all_features is None:
                all_features = cases.copy()[columns_to_select]
            else:
                all_features = pd.concat((all_features, cases[columns_to_select]), axis='index', copy=False)

    # --- Store the data in the DB ---
    with Database() as db:
        store_features_in_db(all_features, chunksize, db.session)

    logger.success('completed')


if __name__ == '__main__':
    create_all_features(
        chunksize=20000,
        n_rows=150000
    )
