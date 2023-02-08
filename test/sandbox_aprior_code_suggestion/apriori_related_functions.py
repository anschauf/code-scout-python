import os
from collections import Counter
from itertools import chain

import pandas as pd
from loguru import logger
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from test.sandbox_aprior_code_suggestion.utils import clean_alt_list


def aprior_algorithms(cases_hospital_all_info, all_codes_df, min_confidence, output_dir):
    num_cases = cases_hospital_all_info.shape[0]
    logger.info(f'Starting apriori rule learning for {num_cases} cases:')
    nrow_original = list()
    case_ids_suggestions = list()
    suggested_codes_pdx = list()
    suggested_codes_times = list()

    hospital = cases_hospital_all_info.iloc[0]['hospital']
    year = cases_hospital_all_info.iloc[0]['dischargeYear']
    if hospital == 'SLI' and year == 2019:
        start_n = 13
    else:
        start_n = 0

    for n in range(start_n, num_cases):
        row_n = cases_hospital_all_info.iloc[n]
        case_id = row_n['id']
        primary_diagnosis = row_n['primaryDiagnosis']
        secondary_diagnoses = row_n['secondaryDiagnoses']
        all_diagnosis = set(secondary_diagnoses)
        all_diagnosis.add(primary_diagnosis)
        procedures = set(row_n['procedures'])
        all_codes = all_diagnosis | procedures
        drg_relevant_icds = set(row_n['drgRelevantDiagnoses'])
        drg_relevant_chops = set(row_n['drgRelevantProcedures'])
        drg_relevant_code = drg_relevant_icds | drg_relevant_chops

        case_info = f'{case_id=}_{hospital=}_{year=}'

        all_codes_df['contain_case_pd'] = all_codes_df['diagnosis_list'].apply(
            lambda x: len(all_diagnosis.intersection(set(x))) > 0)
        all_codes_df['contain_case_procedure'] = all_codes_df['procedure_list'].apply(
            lambda x: len(procedures.intersection(set(x))) > 0)

        subset_for_apriori = all_codes_df[
            (all_codes_df['contain_case_pd'] == True) & (all_codes_df['contain_case_procedure'] == True)]
        # combine icds and chops for each case
        subset_for_apriori['all_code'] = subset_for_apriori['diagnosis_list'] + subset_for_apriori['procedure_list']
        print(subset_for_apriori.shape)

        num_subset = subset_for_apriori.shape[0]
        if num_subset == 0:
            logger.info(f'No useful subsets contain case codes for {n + 1}th {case_info})')
            continue
        elif num_subset <= 200:
            min_support = 0.01
            low_memory = False
            max_len = 3
            # subset_for_apriori.to_csv(os.path.join(output_dir, f'{n + 1}th_case_subsets_{case_info}.csv'))
            # continue
        elif num_subset <= 2000:
            min_support = 0.01
            low_memory = False
            max_len = 4
        else:
            min_support = 0.001
            low_memory = True
            max_len = 4

        # Instantiate transaction encoder and identify unique items in transactions
        encoder = TransactionEncoder()

        # One-hot encode transactions
        subset_icds_list = subset_for_apriori['all_code'].to_list()
        fitted = encoder.fit_transform(subset_icds_list, sparse=True)

        df_icds_onehot = pd.DataFrame.sparse.from_spmatrix(fitted, columns=encoder.columns_)  # seemed to work good

        # Compute frequent itemsets using the Apriori algorithm
        logger.info(f'Computing the {n + 1}/{num_cases} frequent itemset for {case_info} ...')

        # min_support need less when there are more transactions, verse vise
        frequent_itemsets = apriori(df_icds_onehot,
                                    min_support=min_support,
                                    max_len=max_len,
                                    verbose=1,
                                    low_memory=low_memory,
                                    use_colnames=True)

        logger.info(f'{len(frequent_itemsets)} itemsets are generated')

        # define rules based on the metrics confidence
        frequent_itemsets_rules = association_rules(frequent_itemsets,
                                                    min_threshold=min_confidence)
        logger.info \
            (f'{len(frequent_itemsets_rules)} association rules are generated based on {min_confidence=}.')

        # sort the rules based on the confidence and lift
        frequent_itemsets_rules = frequent_itemsets_rules.sort_values(['confidence', 'lift'], ascending=[False, False])

        # change the antecedents and consequents from frozen set to list
        frequent_itemsets_rules['antecedents_ls'] = frequent_itemsets_rules['antecedents'].apply(lambda x: list(x))
        frequent_itemsets_rules['consequents_ls'] = frequent_itemsets_rules['consequents'].apply(lambda x: list(x))

        # filter rule which contains the codes from example cases
        # based on pd and procedure

        frequent_itemsets_rules['contain_drg_relevant_icd'] = frequent_itemsets_rules['antecedents_ls'].apply(
            lambda x: len(set(x).intersection(drg_relevant_icds)) > 0)
        frequent_itemsets_rules['contain_drg_relevant_chops'] = frequent_itemsets_rules['antecedents_ls'].apply(
            lambda x: len(set(x).intersection(drg_relevant_chops)) > 0)

        frequent_itemsets_rules['condition_one'] = frequent_itemsets_rules['antecedents_ls'].apply(
            lambda x: len(set(x).intersection(all_codes)) == 1 and len(x) == 1)
        frequent_itemsets_rules['condition_two'] = frequent_itemsets_rules['antecedents_ls'].apply(
            lambda x: len(set(x).intersection(all_codes)) == 2 and len(x) == 2)
        frequent_itemsets_rules['condition_three'] = frequent_itemsets_rules['antecedents_ls'].apply(
            lambda x: len(set(x).intersection(all_codes)) == 3 and len(x) == 3)

        frequent_itemsets_rules['suggestion_not_contain_case_code'] = frequent_itemsets_rules['consequents_ls'].apply(
            lambda x: (len(set(x).intersection(all_codes)) == 0) if len(x) <= 1 else (
                    len(set(x).intersection(all_codes)) == 1))

        #      save useful rules to check later
        frequent_itemsets_rules_case_condition_one = frequent_itemsets_rules[(
                                                                                     frequent_itemsets_rules[
                                                                                         'condition_one'] == True) & (
                                                                                     frequent_itemsets_rules[
                                                                                         'suggestion_not_contain_case_code'] == True)]
        frequent_itemsets_rules_case_condition_two = frequent_itemsets_rules[(
                                                                                     frequent_itemsets_rules[
                                                                                         'condition_two'] == True) & (
                                                                                     frequent_itemsets_rules[
                                                                                         'suggestion_not_contain_case_code'] == True)]
        frequent_itemsets_rules_case_condition_three = frequent_itemsets_rules[
            (frequent_itemsets_rules['condition_three'] == True) & (
                    frequent_itemsets_rules['suggestion_not_contain_case_code'] == True)]

        frequent_itemsets_rules_case_condition_one_drg = frequent_itemsets_rules_case_condition_one[
            (frequent_itemsets_rules_case_condition_one['contain_drg_relevant_icd'] == True) |
            (frequent_itemsets_rules_case_condition_one['contain_drg_relevant_chops'] == True)]

        frequent_itemsets_rules_case_condition_two_drg = frequent_itemsets_rules_case_condition_two[
            (frequent_itemsets_rules_case_condition_two['contain_drg_relevant_icd'] == True) |
            (frequent_itemsets_rules_case_condition_two['contain_drg_relevant_chops'] == True)]

        frequent_itemsets_rules_case_condition_three_drg = frequent_itemsets_rules_case_condition_three[
            (frequent_itemsets_rules_case_condition_three['contain_drg_relevant_icd'] == True) |
            (frequent_itemsets_rules_case_condition_three['contain_drg_relevant_chops'] == True)]

        num_1 = frequent_itemsets_rules_case_condition_one_drg.shape[0]
        num_2 = frequent_itemsets_rules_case_condition_two_drg.shape[0]
        num_3 = frequent_itemsets_rules_case_condition_three_drg.shape[0]

        frequent_itemsets_rules_case_condition_one_drg.to_csv(
            os.path.join(output_dir, f'{n + 1}th_case_{num_1}_suggestions_{case_info}_condition_one_drg.csv'))
        frequent_itemsets_rules_case_condition_two_drg.to_csv(
            os.path.join(output_dir, f'{n + 1}th_case_{num_2}_suggestions{num_2}_{case_info}_condition_two_drg.csv'))
        frequent_itemsets_rules_case_condition_three_drg.to_csv(
            os.path.join(output_dir, f'{n + 1}th_case_{num_3}_suggestions_{case_info}_condition_three_drg.csv'))
        frequent_itemsets_rules_case_condition_all_drg = pd.concat(
            [frequent_itemsets_rules_case_condition_one_drg, frequent_itemsets_rules_case_condition_two_drg,
             frequent_itemsets_rules_case_condition_three_drg])
        frequent_itemsets_rules_case_condition_all_drg.to_csv(os.path.join(output_dir,
                                                                           f'{n + 1}th_case_{num_3 + num_2 + num_3}_suggestions_{case_info}_condition_all_drg.csv'))

        # Process the suggested codes and save them in the summary dictionary
        # Delete the existed code from suggestions
        frequent_itemsets_rules_case_condition_all_drg['suggestions'] = frequent_itemsets_rules_case_condition_all_drg[
            'consequents_ls'].apply(
            lambda x: list(set(x).difference(all_codes)))

        all_suggested_code = frequent_itemsets_rules_case_condition_all_drg['suggestions'].tolist()
        all_suggested_code = list(chain(*all_suggested_code))
        all_suggested_code_count = Counter(all_suggested_code)
        suggests_string = '|'.join(all_suggested_code_count.keys())

        nrow_original.append(n)
        case_ids_suggestions.append(case_id)
        suggested_codes_pdx.append(suggests_string)
        suggested_codes_times.append(list(all_suggested_code_count.values()))
        logger.info('Generated rules are saved successfully')

    suggests_summary_df = pd.DataFrame.from_records(
        [nrow_original, case_ids_suggestions, suggested_codes_pdx, suggested_codes_times]).transpose()
    suggests_summary_df.columns = ['nrow_original', 'case_ids_suggestions', 'suggested_codes_pdx',
                                   'suggested_codes_times']
    return suggests_summary_df

def combine_apriori_mindbend(dir_apriori_suggestions, mindbend_suggestions, num_condition_code_apriori, output_dir):
    nrow_original = list()
    case_ids_suggestions = list()
    suggested_codes_pdx = list()
    suggested_codes_times = list()
    suggestions_apriori_df = pd.DataFrame()
    for file_name in os.listdir(dir_apriori_suggestions):
        if file_name.startswith('revised_case_'):
            hospital = file_name.split('_')[2]
            year = int(file_name.split('_')[3])
            suggestions_apriori_df = pd.read_csv(os.path.join(dir_apriori_suggestions, file_name))
            suggestions_apriori_df['hospital'] = hospital
            suggestions_apriori_df['year'] = year

        if file_name.endswith(f'{num_condition_code_apriori}.csv'):
            file_name_ls = file_name.replace("'", '').split('=')
            n = file_name_ls[0].split('_')[0]
            case_id = file_name_ls[1].split('_')[0]
            apriori_suggestions_case = pd.read_csv(os.path.join(dir_apriori_suggestions, file_name))
            if apriori_suggestions_case.shape[0] == 0:
                logger.info(f'No apriori suggestions for case {case_id}')
                continue
            mind_bend_suggestions_icds = \
                mindbend_suggestions[mindbend_suggestions['case_id'] == case_id][
                    'mind_bend_suggested_icds'].tolist()[0]
            mind_bend_suggestions_chops = \
                mindbend_suggestions[mindbend_suggestions['case_id'] == case_id][
                    'mind_bend_suggested_chops'].tolist()[0]

            apriori_suggestions_case['consequents_ls'] = apriori_suggestions_case['consequents_ls'].apply(
                clean_alt_list)
            apriori_suggestions_case['suggested_icds'] = apriori_suggestions_case['consequents_ls'].apply(
                lambda x: list(set(x).intersection(set(mind_bend_suggestions_icds))))
            apriori_suggestions_case['suggested_chops'] = apriori_suggestions_case['consequents_ls'].apply(
                lambda x: list(set(x).intersection(set(mind_bend_suggestions_chops))))

            #  apriori suggestions contain either icds or chops from mindbend
            apriori_suggestions_case['suggested_icds_true'] = apriori_suggestions_case['suggested_icds'].apply(
                lambda x: len(x) > 0)
            apriori_suggestions_case['suggested_chops_true'] = apriori_suggestions_case['suggested_chops'].apply(
                lambda x: len(x) > 0)

            combined_suggestions_apriori_mindbend = apriori_suggestions_case[
                (apriori_suggestions_case['suggested_icds_true'] == True) | (
                        apriori_suggestions_case['suggested_chops_true'] == True)]

            combined_suggestions_apriori_mindbend['suggested_codes'] = combined_suggestions_apriori_mindbend[
                                                                           'suggested_icds'] + \
                                                                       combined_suggestions_apriori_mindbend[
                                                                           'suggested_chops']

            num_suggestions = combined_suggestions_apriori_mindbend.shape[0]
            # save all suggestions both from apriori and mindbend
            if num_suggestions >= 1:
                file_path = os.path.join(output_dir,
                                         f'{n}_{case_id}_{num_suggestions}_combined_suggestions_apriori_mindbend_{num_condition_code_apriori}.csv')
                combined_suggestions_apriori_mindbend.to_csv(file_path)
                logger.info(
                    f'Successfully saved {num_suggestions} combined suggestions from mindbend and apriori for {case_id=}')

            else:
                logger.info(
                    f'No combined suggestions from mindbend and apriori for {n} {case_id=}')

            # Format as in code ranking
            all_suggested_code = combined_suggestions_apriori_mindbend['suggested_codes'].tolist()
            all_suggested_code = list(chain(*all_suggested_code))
            all_suggested_code_count = Counter(all_suggested_code)
            suggests_string = '|'.join(all_suggested_code_count.keys())

            nrow_original.append(n)
            case_ids_suggestions.append(case_id)
            suggested_codes_pdx.append(suggests_string)
            suggested_codes_times.append(list(all_suggested_code_count.values()))
            logger.info('Generated rules are saved successfully')

    suggests_summary_df = pd.DataFrame.from_records(
        [nrow_original, case_ids_suggestions, suggested_codes_pdx, suggested_codes_times]).transpose()
    suggests_summary_df.columns = ['nrow_original', 'case_ids_suggestions', 'suggested_codes_pdx',
                                   'suggested_codes_times']

    return suggests_summary_df, suggestions_apriori_df

