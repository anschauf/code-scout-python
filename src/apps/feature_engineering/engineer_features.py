import pandas as pd
from apps.feature_engineering.medications import get_atc_codes
from data_model.feature_engineering import VENTILATION_HOURS_COL, HAS_VENTILATION_HOURS_COL, \
    VENTILATION_HOURS_ADRG_NO_A_COL, EMERGENCY_COL, HAS_HOURS_IN_ICU_COL, HAS_IMC_EFFORT_POINTS_COL, AGE_BINNED_COL, \
    HAS_NEMS_POINTS_COL
from beartype import beartype


@beartype
def engineer_features(cases: pd.DataFrame) -> pd.DataFrame:

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

    return cases
