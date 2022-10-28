import re

import awswrangler as wr
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from src.utils import save_figure_to_pdf_on_s3, __remove_prefix_and_bucket_if_exists


class DurationOfStayLimit:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def __str__(self):
        return f"Duration of stay from {self.lower} till {self.upper} days."

systematic_indexes = [
    # ('s3://swiss-drg/chop/2016/dz-d-14.04.01-chop2016-multilang-01/CHOP 2016 Multilang CSV.csv', 2016),
    ('s3://swiss-drg/chop/2017/dz-d-14.04.01-chop17-sys-01/CHOP2017_Systematisches_Verzeichnis_DE_2016.07.19.csv', 2017, "ISO-8859-1"),
    ('s3://swiss-drg/chop/2018/dz-d-14.04.01-chop18-sys-01/CHOP 2018_Systematisches_Verzeichnis_DE_2017_07_18.csv', 2018, "ISO-8859-1"),
    ('s3://swiss-drg/chop/2019/dz-d-14.04.01-chop19-sys-01/CHOP 2019_Systematisches_Verzeichnis_DE_2018_07_23.csv', 2019, "ISO-8859-1"),
    ('s3://swiss-drg/chop/2020/dz-d-14.04.01-chop20-sys-01/CHOP 2020_Systematisches_Verzeichnis_DE_2019_07_22.csv', 2020, None),
    ('s3://swiss-drg/chop/2021/dz-d-14/CHOP 2021_Systematisches_Verzeichnis_DE_2020_07_21.csv', 2021, None),
]

for filename, year, encoding in systematic_indexes:
    logger.info(f'Currently extracting CHOP limits for year: {year}')

    chops_dictionary = wr.s3.read_csv(filename, dtype='string', sep=';', on_bad_lines='skip', encoding=encoding)

    def till_upper_limit(u): return  DurationOfStayLimit(0, u)
    def less_than_upper_limit(u): return  DurationOfStayLimit(0, u-1)
    def within_range_of_days(l, u): return  DurationOfStayLimit(l, u)
    def from_lower_limit(l): return DurationOfStayLimit(l, 9999999999)
    def more_than_lower_limit(l): return DurationOfStayLimit(l+1, 9999999999)
    regex_translation = [
        ("bis \d+ Behandlungstage", till_upper_limit),
        ("mindestens \d+ bis \d+ Behandlungstage", within_range_of_days),
        ("weniger als \d+ Behandlungstage", less_than_upper_limit),
        ("\d+ und mehr Behandlungstage", from_lower_limit),
        ("\d+ bis \d+ Tage", within_range_of_days),
        ("bis zu \d+ Tagen", till_upper_limit),
        ("\d+ Tage und mehr", from_lower_limit),
        ("innerhalb von \d+ Tagen", till_upper_limit),
        ("über mehr als \d+ Tage", more_than_lower_limit)
    ]
    regexes_compiled = re.compile( '|'.join([r[0] for r in regex_translation]) )

    ind_matches = list()
    limits = list()
    regex_distribution = np.zeros((len(regex_translation),))
    for i, row in enumerate(chops_dictionary.itertuples()):
        description = row.text
        if regexes_compiled.findall(description):
            ind_matches.append(i)
            current_reg_limits = list()
            current_regex_hits = np.zeros((len(regex_translation), ))
            for regex_index, (reg, limit_func) in enumerate(regex_translation):
                current_match = re.compile(reg).findall(description)
                if len(current_match) == 1:
                    current_regex_hits[regex_index] = 1
                    numbers = [int(d) for d in re.compile("\d+").findall(current_match[0])]
                    current_reg_limits.append((current_match, limit_func(*numbers)))

            if len(current_reg_limits) == 1:
                limits.append(current_reg_limits[0][1])
                regex_distribution += current_regex_hits
            else:
                # take longest match
                ind_longest = np.argmax([len(x[0][0]) for x in current_reg_limits])
                limits.append(current_reg_limits[ind_longest][1])
                regex_distribution[ind_longest] += 1

    ind_matches = np.asarray(ind_matches)
    chops_dictionary_duration_of_stay_limits = chops_dictionary.iloc[ind_matches]
    chops_dictionary_duration_of_stay_limits["DoS_lower_limit"] = [x.lower for x in limits]
    chops_dictionary_duration_of_stay_limits["DoS_upper_limit"] = [x.upper for x in limits]
    wr.s3.to_csv(df=chops_dictionary_duration_of_stay_limits, path=filename.replace('.csv', '_DurationOfStayLimits.csv'), index=False)

    plt.figure()
    plt.bar(x=range(len(regex_distribution)), height=regex_distribution)
    plt.xticks(range(len(regex_distribution)), labels=[x[0] for x in regex_translation], rotation=90)
    plt.title(f'Distribution of {int(np.sum(regex_distribution))} regex matches for duration of stay limits')
    plt.ylabel('Frequency')
    plt.tight_layout()
    save_figure_to_pdf_on_s3(plt, 'swiss-drg', __remove_prefix_and_bucket_if_exists(filename).replace('.csv', '_DurationOfStayLimits_distribution.pdf'))
    plt.close()


    # import os
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # ksw2019 = pd.read_csv(os.path.join(current_dir ,'KSW_2019__FAB_Geschlecht_Alter_DtoD_CHOPs.csv')).dropna(subset=['suggested_codes'])
    # ksw2019 = ksw2019.astype({'suggested_codes': 'string'})
    # ksw2019['suggested_codes_split'] = ksw2019['suggested_codes'].apply(lambda x: x.split("""|"""))
    # all_chops_with_limits = [x[1:].replace('.', '') for x in chops_dictionary_duration_of_stay_limits['zcode']]
    #
    # index = list()
    # ranks = list()
    # for i, case in enumerate(ksw2019.itertuples()):
    #     if len(np.intersect1d(case.suggested_codes_split, all_chops_with_limits)) > 0:
    #         index.append(i)
    #         ranks.append([index for index in range(len(case.suggested_codes_split)) if case.suggested_codes_split[index] in all_chops_with_limits])
    #
    # plt.figure()
    # concatenated_ranks = np.concatenate(ranks)
    # plt.hist(concatenated_ranks, bins=np.linspace(np.min(concatenated_ranks), np.max(concatenated_ranks), 100))
    # plt.xlabel('Ranks codes with DoS limits')
    # plt.ylabel('Frequency')
    # plt.tight_layout()
    # plt.savefig(os.path.join(current_dir, f'hist_ranks_codes_with_limits_{year}.pdf'), bbox_inches='tight')
    #
    #
    # plt.figure()
    # concatenated_ranks = np.concatenate(ranks)
    # plt.hist(concatenated_ranks, bins=1000)
    # plt.xlabel('Ranks codes with DoS limits')
    # plt.ylabel('Frequency')
    # plt.tight_layout()
    # plt.xlim([0, 10])
    # plt.savefig(os.path.join(current_dir, f'hist_ranks_codes_with_limits_zoom`{year}.pdf'), bbox_inches='tight')
    # plt.close()
    #
    # print('')
