import re

import awswrangler as wr
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from src.utils import save_figure_to_pdf_on_s3, __remove_prefix_and_bucket_if_exists


class DurationOfStayLimit:
    def __init__(self, lower: int, upper: int):
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

max_limit = 365 * 100
def till_upper_limit(u): return DurationOfStayLimit(0, u)
def less_than_upper_limit(u): return DurationOfStayLimit(0, u - 1)
def within_range_of_days(l, u): return DurationOfStayLimit(l, u)
def from_lower_limit(l): return DurationOfStayLimit(l, max_limit)
def more_than_lower_limit(l): return DurationOfStayLimit(l + 1, max_limit)


# Each tuple contains the regex name, the regex pattern, and the function which can be used to return an instance of
# `DurationOfStayLimit`. The suffix '_t' stands for 'Tage', '_bt' for 'Behandlungstage'
regex_translation = [
    ('till_upper_limit_bt', r"bis (\d+) Behandlungstage", till_upper_limit),
    ('within_range_of_days_bt', r"mindestens (\d+) bis (\d+) Behandlungstage", within_range_of_days),
    ('less_than_upper_limit', r"weniger als (\d+) Behandlungstage", less_than_upper_limit),
    ('from_lower_limit_bt', r"(\d+) und mehr Behandlungstage", from_lower_limit),
    ('within_range_of_days_t', r"(\d+) bis (\d+) Tage", within_range_of_days),
    ('till_upper_limit_t1', r"bis zu (\d+) Tagen", till_upper_limit),
    ('from_lower_limit_t', r"(\d+) Tage und mehr", from_lower_limit),
    ('till_upper_limit_t2', r"innerhalb von (\d+) Tagen", till_upper_limit),
    ('more_than_lower_limit', r"über mehr als (\d+) Tage", more_than_lower_limit)
]

# Assemble the regex patterns in one regex
assembled_regexes = [f'(?P<{r[0]}>{r[1]})' for r in regex_translation]
compiled_concat_regex = re.compile('(' + '|'.join(assembled_regexes) + ')', re.MULTILINE)

# Map the name of each regex pattern to the compiled pattern, which is used to extract the digits
all_compiled_regexes = {r[0]: re.compile(r[1]) for r in regex_translation}

# Map the name of each regex pattern to the translation function
regex_translation_dict = {r[0]: r[2] for r in regex_translation}

# Map each regex pattern to a numerical ID
regex_pattern_to_id = {info[0]: regex_id for regex_id, info in enumerate(regex_translation)}


def run():
    for filename, year, encoding in systematic_indexes:
        logger.info(f'Currently extracting CHOP limits for year: {year}')

        chops_dictionary = wr.s3.read_csv(filename, dtype='string', sep=';', on_bad_lines='skip', encoding=encoding)

        ind_matches = list()
        limits = list()
        regex_distribution = np.zeros((len(regex_translation),))

        for i, row in enumerate(chops_dictionary.itertuples()):
            description = row.text

            # Run all the regexes against the description
            matches = list(compiled_concat_regex.finditer(description))
            if len(matches) == 1:
                ind_matches.append(i)

                # The named group that matches is the one with a non-None value
                matched_groups = [pattern_name
                                  for pattern_name, matched in matches[0].groupdict().items()
                                  if matched is not None]
                if len(matched_groups) > 1:
                    raise ValueError(f"Expected 1 matched group. Found {len(matched_groups)} instead in '{description}'")

                matched_group = matched_groups[0]

                # Extract the numerical values in the match, and the function to apply to them
                numbers_regex = all_compiled_regexes[matched_group]
                matched_numbers_str = numbers_regex.findall(description)
                matched_numbers_int = np.array(matched_numbers_str, dtype=int).flatten()
                limit_func = regex_translation_dict[matched_group]

                # Convert the numerical range to a `DurationOfStayLimit` instance
                dosl = limit_func(*matched_numbers_int)
                limits.append(dosl)

                # Update the counts of occurrence of each pattern
                idx = regex_pattern_to_id[matched_group]
                regex_distribution[idx] += 1

            elif len(matches) > 1:
                raise ValueError(f"Expected up to 1 match per definition. Found {len(matches)} instead in '{description}'")

        ind_matches = np.asarray(ind_matches)
        chops_dictionary_duration_of_stay_limits = chops_dictionary.iloc[ind_matches]
        chops_dictionary_duration_of_stay_limits["DoS_lower_limit"] = [x.lower for x in limits]
        chops_dictionary_duration_of_stay_limits["DoS_upper_limit"] = [x.upper for x in limits]
        wr.s3.to_csv(df=chops_dictionary_duration_of_stay_limits, path=filename.replace('.csv', '_DurationOfStayLimits.csv'), index=False)
        wr.s3.to_csv(df=chops_dictionary_duration_of_stay_limits[['zcode', 'DoS_lower_limit', 'DoS_upper_limit']], path=filename.replace('.csv', '_DurationOfStayLimits_scala.csv'), index=False)

        plt.figure()
        plt.bar(x=range(len(regex_distribution)), height=regex_distribution)
        plt.xticks(range(len(regex_distribution)), labels=[x[0] for x in regex_translation], rotation=90)
        plt.title(f'Distribution of {int(np.sum(regex_distribution))} regex matches for duration of stay limits')
        plt.ylabel('Frequency')
        plt.tight_layout()
        save_figure_to_pdf_on_s3(plt, 'swiss-drg', __remove_prefix_and_bucket_if_exists(filename).replace('.csv', '_DurationOfStayLimits_distribution.pdf'))
        plt.close()

        import os
        import pandas as pd
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filename_chop_suggestions = os.path.join(current_dir, 'HirslandenZurichEncrypted_2019__FAB__DtoD__CHOPs__ranked.csv')
        ksw2019 = pd.read_csv(filename_chop_suggestions).dropna(subset=['suggested_codes'])
        ksw2019 = ksw2019.astype({'suggested_codes': 'string'})
        ksw2019['suggested_codes_split'] = ksw2019['suggested_codes'].apply(lambda x: x.split("""|"""))
        all_chops_with_limits = [x[1:].replace('.', '') for x in chops_dictionary_duration_of_stay_limits['zcode']]

        index = list()
        case_id = list()
        ranks = list()
        for i, case in enumerate(ksw2019.itertuples()):
            if len(np.intersect1d(case.suggested_codes_split, all_chops_with_limits)) > 0:
                index.append(i)
                case_id.append(case.case_id)
                ranks.append([rank for rank in range(len(case.suggested_codes_split)) if case.suggested_codes_split[rank] in all_chops_with_limits])
        pd.DataFrame({
            'case_id': case_id,
            'case_rank': index,
            'index_chops_with_dos_limits': ['|'.join([str(x) for x in ranks[i]]) for i in range(len(ranks))]
        }).to_csv(os.path.join(current_dir, f'chops_with_dos_limits_{os.path.basename(filename_chop_suggestions).replace(".csv", "")}_{year}.csv'), index=False)

        plt.figure()
        concatenated_ranks = np.concatenate(ranks)
        plt.hist(concatenated_ranks, bins=np.linspace(np.min(concatenated_ranks), np.max(concatenated_ranks), 100))
        plt.xlabel('Ranks codes with DoS limits')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(current_dir, f'hist_ranks_codes_with_limits_{os.path.basename(filename_chop_suggestions).replace(".csv", "")}_{year}.pdf'), bbox_inches='tight')


        plt.figure()
        concatenated_ranks = np.concatenate(ranks)
        plt.hist(concatenated_ranks, bins=1000)
        plt.xlabel('Ranks codes with DoS limits')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.xlim([0, 10])
        plt.savefig(os.path.join(current_dir, f'hist_ranks_codes_with_limits_zoom_{os.path.basename(filename_chop_suggestions).replace(".csv", "")}_{year}.pdf'), bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    run()
