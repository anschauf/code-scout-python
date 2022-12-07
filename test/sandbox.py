import pandas as pd

from src.models.revision import Revision
from src.models.sociodemographics import SOCIODEMOGRAPHIC_ID_COL
from src.service.database import Database

with Database() as db:
    revised_cases_query = (
        db.session
        .query(Revision)
        .filter(Revision.revised.is_(True))
    )
    revised_cases = pd.read_sql(revised_cases_query.statement, db.session.bind)

    sociodemo_ids = revised_cases[SOCIODEMOGRAPHIC_ID_COL].values.tolist()

    original_cases_query = (
        db.session
        .query(Revision)
        .filter(Revision.reviewed.is_(False))
        .filter(Revision.sociodemographic_id.in_(sociodemo_ids))
    )
    original_cases = pd.read_sql(original_cases_query.statement, db.session.bind)


    join_key = SOCIODEMOGRAPHIC_ID_COL
    columns_to_compare = ['dos_id', 'mdc', 'mdc_partition', 'drg', 'drg_cost_weight', 'effective_cost_weight', 'pccl', 'raw_pccl', 'supplement_charge', 'supplement_charge_ppu']
    columns_to_select = [join_key] + columns_to_compare
    suffix_revised = '_rev'

    df = pd.merge(
        original_cases[columns_to_select], revised_cases[columns_to_select],
        on=join_key, how='inner', suffixes=('', suffix_revised)
    ).sort_values('effective_cost_weight', ascending=False)

    side_by_side_columns = [join_key]
    for col in columns_to_compare:
        side_by_side_columns.append(col)
        side_by_side_columns.append(f'{col}{suffix_revised}')
    df = df[side_by_side_columns].reset_index(drop=True)

    df2 = df.copy()
    df2['delta_supp_charge'] = df2['supplement_charge_rev'] - df2['supplement_charge']
    df2 = df2[df2['delta_supp_charge'] != 0]
    df2.sort_values('delta_supp_charge', ascending=False, inplace=True)

    print(df2)

    print('')
