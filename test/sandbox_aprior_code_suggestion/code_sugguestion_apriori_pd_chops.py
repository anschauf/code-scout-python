import os
from collections import Counter
from itertools import chain

import pandas as pd
from loguru import logger
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from src import ROOT_DIR
from test.sandbox_aprior_code_suggestion.utils import clean_alt_list
from test.sandbox_aprior_code_suggestion.apriori_related_functions import aprior_algorithms
from test.sandbox_model_case_predictions.data_handler import load_data, load_data_single_file
from test.sandbox_model_case_predictions.utils import get_revised_case_ids


# revised_case_files = ['AA_2017.json', 'AA_2018.json', 'BS_2017.json', 'BS_2018.json',
#                       'BI_2017.json', 'SA_2017.json', 'LI_2017.json', 'LI_2018.json', 'SA_2018.json', 'ST_2018.json',
#                       'ST_2017.json', 'HI_2016.json', 'HI_2018.json', 'HI_2019.json', 'HI_2017.json', 'SLI_2019.json',
#                       'SRRWS_2019.json', 'KSSG_2021.json', 'FT_2019.json', 'USZ_2019.json', 'BI_2018.json', 'KSW_2017.json',
#                       'KSW_2018.json', 'KSW_2019.json', 'KSW_2020.json']

revised_case_files = [ 'SLI_2019.json','SRRWS_2019.json', 'KSSG_2021.json', 'FT_2019.json', 'USZ_2019.json', 'BI_2018.json', 'KSW_2017.json',
                      'KSW_2018.json', 'KSW_2019.json']


revised_case = True
min_confidence = 0.25
# get the hospital and year
dir_data = os.path.join(ROOT_DIR, 'resources', 'data')
REVISED_CASE_IDS_FILENAME = os.path.join(ROOT_DIR, 'resources', 'data', 'revised_case_ids.csv')
all_data = load_data(only_2_rows=True)
revised_cases_in_data = get_revised_case_ids(all_data, REVISED_CASE_IDS_FILENAME, overwrite=False)


# read all icds and chops files
all_diagnosis_df = pd.read_csv(os.path.join(ROOT_DIR, 'test/sandbox_aprior_code_suggestion/all_cases_diagnosis.csv'))
all_procedures_df = pd.read_csv(os.path.join(ROOT_DIR, 'test/sandbox_aprior_code_suggestion/all_cases_procedures.csv'))
all_codes_df = all_diagnosis_df.merge(all_procedures_df, on='revision_id', suffixes=('', '_right'))[
    ['sociodemographic_id', 'revision_id', 'diagnosis', 'procedure']]

# filter out the data contain both pprimary diagonosis and procedures
all_codes_df['diagnosis_list'] = all_codes_df['diagnosis'].apply(clean_alt_list)
all_codes_df['procedure_list'] = all_codes_df['procedure'].apply(clean_alt_list)


if revised_case:
    for file_name in revised_case_files:
        hospital_year = file_name.replace('.json', '')
        hospital = hospital_year.split('_')[0]
        year = int(hospital_year.split('_')[1])
        folder_name = f'revised_case_{hospital_year}_apriori_{min_confidence=}'
        output_dir = os.path.join(ROOT_DIR, 'results/aprior_code_suggestions', folder_name)

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        data_single_hospital = load_data_single_file(dir_data=dir_data, file_name=file_name)
        revised_case_hospital = revised_cases_in_data[
        (revised_cases_in_data['hospital'] == hospital) & (revised_cases_in_data['dischargeYear'] == year) & (
                revised_cases_in_data['is_revised'] == 1)]
        cases_hospital_all_info = revised_case_hospital.merge(data_single_hospital, on='id', suffixes=("", "_right"))

        suggests_summary_df = aprior_algorithms(cases_hospital_all_info, all_codes_df, min_confidence, output_dir)
        suggests_summary_df.to_csv(os.path.join(output_dir, f'revised_case_{hospital}_{year}_suggestions_summary.csv'))


else:
    topn = 100
    clinic = 'm100_200'
    file_name = 'KSW_2020.json'
    hospital_year = 'KSW_2020'
    data_single_hospital = load_data_single_file(dir_data=dir_data, file_name=file_name)
    revised_cases_in_data = get_revised_case_ids(data_single_hospital, REVISED_CASE_IDS_FILENAME, overwrite=False)

    folder_name = f'top{topn}_cases_{hospital_year}_apriori_{min_confidence=}_{clinic}'
    output_dir = os.path.join(ROOT_DIR, 'results/aprior_code_suggestions', folder_name)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    # case_id for top 500 cases from random forest
    case_rf = pd.read_csv(os.path.join(ROOT_DIR,
                                       'test/sandbox_aprior_code_suggestion/n-estimator-1000_max-depth-9_min-sample-leaf-400_min-samples-split-1_KSW-2020.csv'),
                          sep=";")
    case_rf.sort_values(by='UpcodingConfidenceScore', inplace=True, ascending=False)
    # exclude get the revised and reviewed cases
    cases_in_data_hospital = revised_cases_in_data[
        (revised_cases_in_data['hospital'] == 'KSW') & (revised_cases_in_data['dischargeYear'] == 2020)]
    revised_reviewed_cases_hospital = cases_in_data_hospital[
        (cases_in_data_hospital['is_reviewed'] == 1) | (cases_in_data_hospital['is_revised'] == 1)]

    # only pick M100 and M200
    cases_in_data_hospital_m100_m200 = data_single_hospital[
        (data_single_hospital['Hauptkostenstelle'] == 'M100') | (data_single_hospital['Hauptkostenstelle'] == 'M200')]
    cases_in_data_hospital_m100_m200_case_ids = cases_in_data_hospital_m100_m200['id'].tolist()

    revised_reviewed_cases_hospital_case_ids = revised_reviewed_cases_hospital['id'].tolist()
    case_ids_rf = case_rf['CaseId'].astype(str).tolist()

    # get case_ids (not reviewed and not reivsed)
    case_ids_not_reviewed_revised = [case_id for case_id in case_ids_rf if
                                     (case_id not in revised_reviewed_cases_hospital_case_ids)]
    # get case_ids from m100 and m200
    case_ids_m100_m200 = [case_id for case_id in case_ids_not_reviewed_revised if
                          (case_id in cases_in_data_hospital_m100_m200_case_ids)]

    # pick up top n for suggestions
    case_ids = case_ids_m100_m200[:topn]
    cases_hospital_rf_topn = cases_in_data_hospital[cases_in_data_hospital['id'].isin(case_ids)]
    cases_hospital_all_info = cases_hospital_rf_topn.merge(data_single_hospital, on='id', suffixes=("", "_right"))
    suggests_summary_df = aprior_algorithms(cases_hospital_all_info, all_codes_df, min_confidence, output_dir)
    suggests_summary_df.to_csv(os.path.join(output_dir, f'top{topn}_{hospital_year}_apriori_{min_confidence=}_{clinic}_suggestions_summary.csv'))

# all columns name cases_hospital_all_info
# index, id, hospital, dischargeYear, is_revised, is_reviewed, drg_old, drg_new, cw_old, cw_new, pccl_old, pccl_new,
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
