from dataclasses import dataclass
from os.path import join

import numpy as np
import pandas as pd

from src import ROOT_DIR

filename_results = join(ROOT_DIR, 'results/missing_additional_chops_multilang_reporting_new_implementation/2019/2023-02-15_13:26:12/upcodeable_codes.csv')
results = pd.read_csv(filename_results, dtype={'triggering_chop': 'string',
                                               'chop_to_replace': 'string',
                                               'chop_to_insert': 'string'})


def unique_list(x):
    return np.unique(x).tolist()


results_aggregated_results = results.groupby(by=['triggering_chop', 'chop_to_replace'], as_index=False).agg(list)


@dataclass(frozen=True)
class ReplacementRule:
    triggering_code: str
    code_to_replace: str
    code_to_add: list[str]
    count_cases: list[int]
    old_drg: list[str]
    target_drg: list[str]


replacement_rule = list()
for code_pairs in results_aggregated_results.itertuples():
    codes_added_unique, codes_added_counts = np.unique(code_pairs.chop_to_insert, return_counts=True)
    replacement_rule.append(ReplacementRule(triggering_code=code_pairs.triggering_chop, code_to_replace=code_pairs.chop_to_replace, code_to_add=codes_added_unique.tolist(), count_cases=codes_added_counts.tolist(), old_drg=np.unique(code_pairs.drg).tolist(), target_drg=np.unique(code_pairs.new_drg).tolist()))


pd.DataFrame({
    'triggering_code': [x.triggering_code for x in replacement_rule],
    'code_to_replace': [x.code_to_replace for x in replacement_rule],
    'code_to_add': ['|'.join([str(y) for y in x.code_to_add]) if len(x.code_to_add) > 1 else str(x.code_to_add[0]) for x in replacement_rule],
    'count_cases': ['|'.join([str(y) for y in x.count_cases]) if len(x.count_cases) > 1 else str(x.count_cases[0]) for x in replacement_rule],
    'old_drg': ['|'.join([str(y) for y in x.old_drg]) if len(x.old_drg) > 1 else str(x.old_drg[0]) for x in replacement_rule],
    'target_drg': ['|'.join([str(y) for y in x.target_drg]) if len(x.target_drg) > 1 else str(x.target_drg[0]) for x in replacement_rule]
}, dtype='string').to_csv(filename_results.replace('.csv', '_rule_summary.csv'), index=False)

