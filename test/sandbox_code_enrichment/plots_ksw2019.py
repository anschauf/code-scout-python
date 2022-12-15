from os.path import join

import awswrangler as wr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.general_utils import save_figure_to_pdf_on_s3

THRESHOLD_ENRICHMENT = 0.05
bucket = 'code-scout'
s3_prefix = 's3://'
filename = 'code_enrichment/example_ksw2019'
dir_data = join(f'{s3_prefix}{bucket}', filename)
dir_output = dir_data

# load ground truth data from revised cases
ground_truth = wr.s3.read_csv(join(dir_data, 'CodeScout_GroundTruthforPerformanceMeasuring.csv'))
ground_truth = ground_truth[(ground_truth['Hospital'] == 'Kantonsspital Winterthur') & (ground_truth['Year'] == 2019)]
ground_truth_diags_added = np.unique(np.concatenate([x.split('|') for x in ground_truth['ICD_added'].values if isinstance(x, str)]))
ground_truth_chops_added = np.concatenate([x.split('|') for x in ground_truth['CHOP_added'].values if isinstance(x, str)])
ground_truth_chops_added = np.unique(np.asarray([x.split(':')[0].replace('.', '') for x in ground_truth_chops_added]))

# load enrichment data
enrichment_diags = wr.s3.read_csv(join(dir_data, 't-test_diagnoses.csv'))
enrichment_chops = wr.s3.read_csv(join(dir_data, 't-test_chops.csv'))

# plot barplot diagnoses
count_underrepresented_diags = 0
count_overrepresented_diags = 0
count_not_enriched_diags = 0
list_diags_added_and_enriched = list()
list_enrichment_diags = list()
for diag in ground_truth_diags_added:
    ind_diag = np.where(enrichment_diags['diag'].values == diag)[0]
    if len(ind_diag) == 1:
        if enrichment_diags['pval_adj_fdr'].values[ind_diag] < THRESHOLD_ENRICHMENT:
            list_diags_added_and_enriched.append(diag)
            if enrichment_diags['mean_all'].values[ind_diag] < enrichment_diags['mean_ksw2019'].values[ind_diag]:
                count_overrepresented_diags += 1
                list_enrichment_diags.append(1)
            else:
                count_underrepresented_diags += 1
                list_enrichment_diags.append(-1)
        else:
            count_not_enriched_diags += 1

wr.s3.to_csv(pd.DataFrame({
    'diag': list_diags_added_and_enriched,
    'enrichment': list_enrichment_diags
}), join(dir_output, 'enriched_revised_diags.csv'), index=False)
diags_enrichment_counts = [count_not_enriched_diags, count_underrepresented_diags, count_overrepresented_diags]
data_diags = pd.DataFrame({
    'Counts': diags_enrichment_counts,
    'Enrichment Type': ['Not Enriched', 'Underrepresented', 'Overrepresented']
})

plt.figure()
sns.barplot(data=data_diags, x='Enrichment Type', y='Counts')
plt.yticks(diags_enrichment_counts, diags_enrichment_counts)
save_figure_to_pdf_on_s3(plt, bucket=bucket, filename=join(filename, 'barplot_enrichment_diagnoses.pdf'))
plt.close()

# plot barplot chops
count_underrepresented_chops = 0
count_overrepresented_chops = 0
count_not_enriched_chops = 0
list_chops_added_and_enriched = list()
list_enrichment_chops = list()
for chop in ground_truth_chops_added:
    ind_chop = np.where(enrichment_chops['chops'].values == chop)[0]
    if len(ind_chop) == 1:
        if enrichment_chops['pval_adj_fdr'].values[ind_chop] < THRESHOLD_ENRICHMENT:
            list_chops_added_and_enriched.append(chop)
            if enrichment_chops['mean_all'].values[ind_chop] < enrichment_chops['mean_ksw2019'].values[ind_chop]:
                count_overrepresented_chops += 1
                list_enrichment_chops.append(1)
            else:
                count_underrepresented_chops += 1
                list_enrichment_chops.append(-1)
        else:
            count_not_enriched_chops += 1

wr.s3.to_csv(pd.DataFrame({
    'chop': list_chops_added_and_enriched,
    'enrichment': list_enrichment_chops
}), join(dir_output, 'enriched_revised_chops.csv'), index=False)
chops_enrichment_counts = [count_not_enriched_chops, count_underrepresented_chops, count_overrepresented_chops]
data_chops = pd.DataFrame({
    'Counts': chops_enrichment_counts,
    'Enrichment Type': ['Not Enriched', 'Underrepresented', 'Overrepresented']
})

plt.figure()
sns.barplot(data=data_chops, x='Enrichment Type', y='Counts')
plt.yticks(chops_enrichment_counts, chops_enrichment_counts)
save_figure_to_pdf_on_s3(plt, bucket=bucket, filename=join(filename, 'barplot_enrichment_chops.pdf'))
plt.close()

print('')
