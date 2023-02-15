from dataclasses import dataclass
from os.path import join

import numpy as np
import pandas as pd

from src import ROOT_DIR

filename_results = join(ROOT_DIR, 'results/code_enrichment_analysis/Kantonsspital Winterthur_2019/2023-02-15_07:54:04/upcodeable_codes.csv')
results = pd.read_csv(filename_results)

results_aggregated_enriched_code = results.groupby(by='enriched_code', as_index=False).agg(list)


@dataclass(frozen=True)
class ReplacementRule:
    code_to_replace: str
    mean_all: float
    mean_test_hospital: float
    code_to_add: list[str]
    count_cases: list[int]
    old_drg: list[str]
    target_drg: list[str]


replacement_rule = list()
for enriched_code in results_aggregated_enriched_code.itertuples():
    codes_added_unique, codes_added_counts = np.unique(enriched_code.codes_to_add, return_counts=True)
    replacement_rule.append(ReplacementRule(code_to_replace=enriched_code.enriched_code, mean_all=enriched_code.mean_all[0], mean_test_hospital=enriched_code.mean_test_hospital[0], code_to_add=codes_added_unique.tolist(), count_cases=codes_added_counts.tolist(), old_drg=np.unique(enriched_code.drg).tolist(), target_drg=np.unique(enriched_code.new_drg).tolist()))


pd.DataFrame({
    'code_to_replace': [x.code_to_replace for x in replacement_rule],
    'mean_all': [x.mean_all for x in replacement_rule],
    'mean_test_hospital': [x.mean_test_hospital for x in replacement_rule],
    'code_to_add': ['|'.join([str(y) for y in x.code_to_add]) if len(x.code_to_add) > 1 else str(x.code_to_add[0]) for x in replacement_rule],
    'count_cases': ['|'.join([str(y) for y in x.count_cases]) if len(x.count_cases) > 1 else str(x.count_cases[0]) for x in replacement_rule],
    'old_drg': ['|'.join([str(y) for y in x.old_drg]) if len(x.old_drg) > 1 else str(x.old_drg[0]) for x in replacement_rule],
    'target_drg': ['|'.join([str(y) for y in x.target_drg]) if len(x.target_drg) > 1 else str(x.target_drg[0]) for x in replacement_rule]
}).to_csv(filename_results.replace('.csv', '_rule_summary.csv'), index=False)

