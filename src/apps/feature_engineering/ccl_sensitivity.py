import pandas as pd
from beartype import beartype
from loguru import logger

from apps.feature_engineering.medications import get_atc_codes
from src.data_model.feature_engineering import DELTA_CCL_TO_NEXT_PCCL_COL, VENTILATION_HOURS_COL, \
    HAS_VENTILATION_HOURS_COL, VENTILATION_HOURS_ADRG_NO_A_COL, EMERGENCY_COL, HAS_HOURS_IN_ICU_COL, \
    HAS_IMC_EFFORT_POINTS_COL, AGE_BINNED_COL, HAS_NEMS_POINTS_COL
from src.revised_case_normalization.notebook_functions.global_configs import PCCL_COL, RAW_PCCL_COL


@beartype
def calculate_delta_pccl(cases: pd.DataFrame, *, delta_value_for_max: float = 0.0) -> pd.DataFrame:
    def _calculate_delta_pccl(row):
        # The max PCCL from 2022 is 6 instead of 4
        if row['discharge_year'] <= 2021:
            max_pccl = 4
        else:
            max_pccl = 6

        current_pccl = row[PCCL_COL]
        if current_pccl == max_pccl:
            row[DELTA_CCL_TO_NEXT_PCCL_COL] = delta_value_for_max

        else:
            raw_pccl = row[RAW_PCCL_COL]
            target_pccl = current_pccl + 1
            target_raw_pccl = target_pccl - 0.49
            row[DELTA_CCL_TO_NEXT_PCCL_COL] = target_raw_pccl - raw_pccl

        return row

    logger.info('Calculating the delta CCL to reach the next PCCL value ...')
    cases = cases.apply(_calculate_delta_pccl, axis=1)
    return cases



