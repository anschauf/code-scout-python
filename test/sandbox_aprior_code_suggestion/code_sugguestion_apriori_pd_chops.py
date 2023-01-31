import os

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from src import ROOT_DIR
from test.sandbox_model_case_predictions.data_handler import load_data_single_file
from test.sandbox_model_case_predictions.utils import get_revised_case_ids

hospital = "Kantonsspital Winterthur"
file_name = 'KSW_2020.json'
year = 2020
# get the hospital and year
dir_data = os.path.join(ROOT_DIR, 'resources', 'data')
REVISED_CASE_IDS_FILENAME = os.path.join(ROOT_DIR, 'resources', 'data', 'revised_case_ids.csv')

data_ksw2020 = load_data_single_file(dir_data=dir_data, file_name=file_name)
revised_cases_in_data = get_revised_case_ids(data_ksw2020, REVISED_CASE_IDS_FILENAME, overwrite=False)

# cases need to give suggestions
# revised_cases
revised_case_ksw2020 = revised_cases_in_data[
    (revised_cases_in_data['hospital'] == 'KSW') & (revised_cases_in_data['dischargeYear'] == 2020) & (
            revised_cases_in_data['is_revised'] == 1)]
revised_case_ksw2020_all_info = revised_case_ksw2020.merge(data_ksw2020, on='id', suffixes=("", "_right"))
#
#  top 500 from RF case ranking
# read the case ranking from rf



all_diagnosis_df = pd.read_csv(os.path.join(ROOT_DIR, 'test/sandbox_aprior_code_suggestion/all_cases_diagnosis.csv'))
all_procedures_df = pd.read_csv(os.path.join(ROOT_DIR, 'test/sandbox_aprior_code_suggestion/all_cases_procedures.csv'))


# min_support = 1.0/case_with_case_pd.shape[0]

# min_support = 0.001
# clear the code and save as a list
def clean_alt_list(list_):
    list_ = list_.strip('[ ]')
    list_ = list_.replace("'", '')
    list_ = list_.split(', ')

    return list_


# case_need to get the suggestions
# example_ksw_2020_revised_cases_before_revision = pd.read_csv(os.path.join(ROOT_DIR, 'test/sandbox_aprior_code_suggestion/example_ksw_2020_revised_cases_before_revision.csv'))
# revised_case_ksw_2020_all_info['secondary_diagnoses'] = revised_case_ksw_2020_all_info['secondary_diagnoses'].apply(clean_alt_list)
# revised_case_ksw_2020_all_info.to_csv('revised_case_ksw_2020_all_info.csv')

# all columns name, index, id, hospital, dischargeYear, is_revised, is_reviewed, drg_old, drg_new, cw_old, cw_new, pccl_old, pccl_new,
# diagnoses_added, procedures_added, index_right, entryDate, exitDate, birthDate, leaveDays, ageYears, ageDays, admissionWeight,
# gender, grouperAdmissionCode, grouperDischargeCode, gestationAge, durationOfStay, hoursMechanicalVentilation, primaryDiagnosis,
# secondaryDiagnoses, procedures, medications, ageFlag, admissionWeightFlag, genderFlag, grouperAdmissionCodeFlag, grouperDischargeCodeFlag,
# durationOfStayFlag, hoursMechanicalVentilationFlag, gestationAgeFlag, isNewborn, durationOfStayCaseType, grouperStatus, drg, mdc,
# mdcPartition, pccl, rawPccl, diagnosesForPccl, drgRelevantDiagnoses, drgRelevantProcedures, diagnosesExtendedInfo, proceduresExtendedInfo,
# drgCostWeight, effectiveCostWeight, drgRelevantGlobalFunctions, supplementCharges, supplementChargesPPU, supplementChargePerCode,
# AnonymerVerbindungskode, ArtDesSGIScore, AufenthaltIntensivstation, AufenthaltNachAustritt, AufenthaltsKlasse, AufenthaltsortVorDemEintritt,
# BehandlungNachAustritt, Eintrittsart, EinweisendeInstanz, EntscheidFuerAustritt, ErfassungDerAufwandpunkteFuerIMC, Hauptkostenstelle,
# HauptkostentraegerFuerGrundversicherungsleistungen, NEMSTotalAllerSchichten, WohnortRegion, VectorizedCodes, IsCaseBelowPcclSplit,
# NumDrgRelevantDiagnoses, NumDrgRelevantProcedures, GeburtsdatumDerMutter, KindVitalstatus, KongenitaleMissbildungen, Mehrlingsgeburten,
# AlterDerMutter, hospital_right, dischargeYear_right

# n = 0
for n in range(0, revised_case_ksw2020_all_info.shape[0]):
    row_n = revised_case_ksw2020_all_info.iloc[n]

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

    all_codes_df = all_diagnosis_df.merge(all_procedures_df, on='revision_id', suffixes=('', '_right'))[
        ['sociodemographic_id', 'revision_id', 'diagnosis', 'procedure']]

    # filter out the data contain both pprimary diagonosis and procedures
    all_codes_df['diagnosis_list'] = all_codes_df['diagnosis'].apply(clean_alt_list)
    all_codes_df['procedure_list'] = all_codes_df['procedure'].apply(clean_alt_list)

    all_codes_df['contain_case_pd'] = all_codes_df['diagnosis_list'].apply(
        lambda x: len(all_diagnosis.intersection(set(x))) > 0)
    all_codes_df['contain_case_procedure'] = all_codes_df['procedure_list'].apply(
        lambda x: len(procedures.intersection(set(x))) > 0)

    subset_for_apriori = all_codes_df[
        (all_codes_df['contain_case_pd'] == True) & (all_codes_df['contain_case_procedure'] == True)]
    # combine icds and chops for each case
    subset_for_apriori['all_code'] = subset_for_apriori['diagnosis_list'] + subset_for_apriori['procedure_list']
    print(subset_for_apriori.shape)

    # Instantiate transaction encoder and identify unique items in transactions
    encoder = TransactionEncoder()

    # One-hot encode transactions
    subset_icds_list = subset_for_apriori['all_code'].to_list()
    fitted = encoder.fit_transform(subset_icds_list, sparse=True)

    df_icds_onehot = pd.DataFrame.sparse.from_spmatrix(fitted, columns=encoder.columns_)  # seemed to work good
    df_icds_onehot.shape

    # Compute frequent itemsets using the Apriori algorithm

    frequent_itemsets = apriori(df_icds_onehot,
                                min_support=0.001,
                                max_len=4,
                                verbose=1,
                                low_memory=True,
                                use_colnames=True)

    # Print a preview of the frequent itemsets
    print(len(frequent_itemsets))

    # define rules based on the metrics support

    frequent_itemsets_rules = association_rules(frequent_itemsets,
                                                metric="support",
                                                min_threshold=0.0015)
    # sort the rules based on the confidence and lift
    frequent_itemsets_rules = frequent_itemsets_rules.sort_values(['confidence', 'lift'], ascending=[False, False])

    # change the antecedents and consequents from frozen set to list
    frequent_itemsets_rules['antecedents_ls'] = frequent_itemsets_rules['antecedents'].apply(lambda x: list(x))
    frequent_itemsets_rules['consequents_ls'] = frequent_itemsets_rules['consequents'].apply(lambda x: list(x))

    # filter rule which contains the codes from example cases
    # based on pd and procedure

    frequent_itemsets_rules['contain_case_code'] = frequent_itemsets_rules['antecedents_ls'].apply(
        lambda x: len(set(x).intersection(all_codes)) > 0)
    frequent_itemsets_rules['contain_drg_relevant_icd'] = frequent_itemsets_rules['antecedents_ls'].apply(
        lambda x: len(set(x).intersection(drg_relevant_icds)) > 0)
    frequent_itemsets_rules['contain_drg_relevant_chops'] = frequent_itemsets_rules['antecedents_ls'].apply(
        lambda x: len(set(x).intersection(drg_relevant_chops)) > 0)
    frequent_itemsets_rules['contain_drg_relevant_code'] = frequent_itemsets_rules['antecedents_ls'].apply(
        lambda x: len(set(x).intersection(drg_relevant_code)) > 0)

    frequent_itemsets_rules['condition_one'] = frequent_itemsets_rules['antecedents_ls'].apply(
        lambda x: len(set(x).intersection(all_codes)) == 1)
    frequent_itemsets_rules['condition_two'] = frequent_itemsets_rules['antecedents_ls'].apply(
        lambda x: len(set(x).intersection(all_codes)) == 2)
    frequent_itemsets_rules['condition_three'] = frequent_itemsets_rules['antecedents_ls'].apply(
        lambda x: len(set(x).intersection(all_codes)) == 3)

    frequent_itemsets_rules['case_rule_result'] = frequent_itemsets_rules['consequents_ls'].apply(
        lambda x: len(set(x).intersection(all_codes)) == 0)

    #      save useful rules to check later
    frequent_itemsets_rules_case = frequent_itemsets_rules[frequent_itemsets_rules['contain_case_code'] == True]
    frequent_itemsets_rules_case_condition_two_plus = frequent_itemsets_rules[frequent_itemsets_rules['condition_two']]
    frequent_itemsets_rules_case_condition_two_plus.to_csv(f'suggestions_{case_info}_condition_two_plus.csv')

