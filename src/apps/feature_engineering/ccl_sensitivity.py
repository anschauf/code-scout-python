import pandas as pd
from beartype import beartype
from loguru import logger

from src.data_model.feature_engineering import DELTA_CCL_TO_NEXT_PCCL_COL
from src.revised_case_normalization.notebook_functions.global_configs import PCCL_COL, RAW_PCCL_COL


@beartype
def calculate_delta_pccl(cases: pd.DataFrame, *, delta_value_for_max: float = 0.0) -> pd.DataFrame:

    """As PCCL levels as calculated by the grouper are integers of a scale between 0 and 4 (from 2022 on: 0 to 6),
    the real/raw PCCL values are rounded.

    This function calculates the raw PCCL points needed for a case to receive a higher PCCL level.

    e.g.(based on DRG system until 2021): A case with a raw PCCL level of 2.5 would be rounded up to a PCCL of 3,
    whereas a raw PCCL level of 2.4 would be rounded down to 2.

    Information on the calculation of PCCL levels can be found in the "definitions Handbuch of SwissDRG"
    https://www.swissdrg.org/de/akutsomatik/swissdrg-system-1102022/definitionshandbuch (link here of Version 11.0, 2022)
    """

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



