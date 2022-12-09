import pandas as pd
from apps.feature_engineering.medications import get_atc_codes
from data_model.feature_engineering import VENTILATION_HOURS_COL, HAS_VENTILATION_HOURS_COL, \
    VENTILATION_HOURS_ADRG_NO_A_COL, EMERGENCY_COL, HAS_HOURS_IN_ICU_COL, HAS_IMC_EFFORT_POINTS_COL, AGE_BINNED_COL, \
    HAS_NEMS_POINTS_COL
from beartype import beartype


@beartype
def engineer_features(row: pd.DataFrame) -> pd.DataFrame:
    # Ventilation hours
    row[VENTILATION_HOURS_COL] = row[VENTILATION_HOURS_COL].fillna(0)
    row[HAS_VENTILATION_HOURS_COL] = row[VENTILATION_HOURS_COL] > 0

    # 'has_ventilation_hours' (boolean) * aDRG not starting with "A"
    row['adrg_no_A'] = row['adrg'].str[0]
    row['adrg_no_A'] = row['adrg_no_A'] != "A"
    row[VENTILATION_HOURS_ADRG_NO_A_COL] = row['adrg_no_A'] * row[HAS_VENTILATION_HOURS_COL]

    # Emergency Boolean
    row[EMERGENCY_COL] = row['admission_type'] == "1"

    # Hours in ICU Boolean
    row[HAS_HOURS_IN_ICU_COL] = row['hours_in_icu'] > 0

    # Has IMC effort points boolean
    row[HAS_IMC_EFFORT_POINTS_COL] = row['imc_effort_points'] > 0

    # create age bins
    age_bins = [0, 1, 2, 5, 9, 10, 15, 16, 19, 20, 29, 30, 40, 49, 50, 59, 60, 69, 70, 79, 80, 89, 90, 99, 100, 120]
    row[AGE_BINNED_COL] = pd.cut(row['age_years'], age_bins)
    row[AGE_BINNED_COL] = row[AGE_BINNED_COL].astype(str).str.replace('nan', '0')

    # Nems Boolean
    row[HAS_NEMS_POINTS_COL] = row['nems_total'] > 0

    # Weekday from date admission
    row['admission_date'] = pd.to_datetime(row['admission_date'])
    row['admission_date_weekday'] = row['admission_date'].dt.day_name()

    # Month of admission date
    row['admission_date_month'] = row['admission_date'].dt.month_name()

    # Weekday from discharge date
    row['discharge_date'] = pd.to_datetime(row['discharge_date'])
    row['discharge_date_weekday'] = row['discharge_date'].dt.day_name()

    # Month of admission date
    row['discharge_date_month'] = row['discharge_date'].dt.month_name()

    row = get_atc_codes(row)

    return row
