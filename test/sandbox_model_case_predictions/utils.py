from datetime import datetime
from itertools import chain, combinations

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MultiLabelBinarizer


def get_list_of_all_predictors(data):
    # define predictors from data
    list_X = list()
    list_X_labels = list()
    list_x_labels_predictor_title = list()

    # IMC effort points
    list_X.append(data['ErfassungDerAufwandpunkteFuerIMC'].values.reshape(-1, 1))
    list_X_labels.append(['erfassung_der_aufwandpunkte_fuer_IMC'])
    list_x_labels_predictor_title.append(['erfassung_der_aufwandpunkte_fuer_IMC'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # Ventilation hours
    list_X.append(data['hoursMechanicalVentilation'].values.reshape(-1, 1))
    list_X_labels.append(['hours_mechanical_ventilation'])
    list_x_labels_predictor_title.append(['hours_mechanical_ventilation'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # Ventilation hours boolean
    ventilation_hours_boolean = np.asarray([int(x) for x in (data['hoursMechanicalVentilation'].apply(__int_to_bool))]).reshape(-1, 1)
    list_X.append(ventilation_hours_boolean)
    list_X_labels.append(['hours_mechanical_ventilation_boolean'])
    list_x_labels_predictor_title.append(['hours_mechanical_ventilation_boolean'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # Ventilation hours bins
    # left out, because not much data for ventilation hours

    # Ventilation hours (as a boolean) multiplied by ${DRG does not start with “A”}
    drg_starts_with_a = np.asarray([int(x) for x in data['drg'].apply(lambda x: True if (x.startswith('A')) else False).values]).reshape(-1, 1)
    list_X.append(ventilation_hours_boolean * drg_starts_with_a)
    list_X_labels.append(['ventilation_hours_boolean_multiplied_with_DRG_starts_with_A'])
    list_x_labels_predictor_title.append(['ventilation_hours_boolean_multiplied_with_DRG_starts_with_A'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # Notfall/ Emergency boolean
    emergency = np.asarray([1 if (x=='1') else 0 for x in data['Eintrittsart'].values]).reshape(-1,1)
    list_X.append(emergency)
    list_X_labels.append(['emergency_boolean'])
    list_x_labels_predictor_title.append(['emergency_boolean'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # Admission type
    admission_type, admisstion_type_label, _ = categorize_variable(data, 'Eintrittsart')
    list_X.append(admission_type)
    list_X_labels.append(admisstion_type_label)
    list_x_labels_predictor_title.append(['admission_type'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # discharge type
    discharge_type, discharge_type_label, _ = categorize_variable(data, 'EntscheidFuerAustritt')
    list_X.append(discharge_type)
    list_X_labels.append(discharge_type_label)
    list_x_labels_predictor_title.append(['discharge_type'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # hours in ICU
    list_X.append(data['AufenthaltIntensivstation'].values.reshape(-1,1))
    list_X_labels.append(['aufenthalt_intensivstation'])
    list_x_labels_predictor_title.append(['aufenthalt_intensivstation'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # hours in ICU boolean
    hours_in_icu_boolean = np.asarray([int(x) for x in data['AufenthaltIntensivstation'].apply(__int_to_bool).values]).reshape(-1,1)
    list_X.append(hours_in_icu_boolean)
    list_X_labels.append(['aufenthalt_intensivstation_boolean'])
    list_x_labels_predictor_title.append(['aufenthalt_intensivstation_boolean'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # Does ADRG has PCCL-split boolean
    #TODO need infos

    # CCL sensitivity
    #TODO ask paolo for code

    # NEMS boolean
    nems_boolean = np.asarray([int(x) for x in data['NEMSTotalAllerSchichten'].apply(__int_to_bool).values]).reshape(-1,1)
    list_X.append(nems_boolean)
    list_X_labels.append(['nems_total_aller_schichten_boolean'])
    list_x_labels_predictor_title.append(['nems_total_aller_schichten_boolean'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # NEMS
    list_X.append(data['NEMSTotalAllerSchichten'].values.reshape(-1,1))
    list_X_labels.append(['nems_total_aller_schichten'])
    list_x_labels_predictor_title.append(['nems_total_aller_schichten'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # IMC effort points boolean
    #TODO check if it has a true boolean with all data
    imc_effort_points_boolean = np.asarray([int(x) for x in data['ErfassungDerAufwandpunkteFuerIMC'].apply(__int_to_bool).values]).reshape(-1,1)
    list_X.append(imc_effort_points_boolean)
    list_X_labels.append(['erfassung_der_aufwandpunkte_fuer_imc'])
    list_x_labels_predictor_title.append(['erfassung_der_aufwandpunkte_fuer_imc'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # Medication ATC-code
    #TODO think about reducing ATC codes by for example just taking first 3 letters
    # e.g. see here: https://www.wido.de/publikationen-produkte/arzneimittel-klassifikation/
    data['medications_atc'] = data['medications'].apply(lambda all_meds: set([x.split(':')[0] for x in all_meds]))
    medication_atc_codes_binary, medication_atc_codes_labels, _ = categorize_variable(data, 'medications_atc')
    list_X.append(medication_atc_codes_binary)
    list_X_labels.append(medication_atc_codes_labels)
    list_x_labels_predictor_title.append(['medications'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # Medication kind
    data['medications_kind'] = data['medications'].apply(lambda all_meds: set([x.split(':')[2] for x in all_meds]))
    medication_kind_binar, medication_kind_labels, _ = categorize_variable(data, 'medications_kind')
    list_X.append(medication_kind_binar)
    list_X_labels.append(medication_kind_labels)
    list_x_labels_predictor_title.append(['medication_kind'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # medication rate
    #TODO difficult to compare with different units

    # medication frequency
    #TODO not sure how to get the frequency

    # month admission
    data['month_admission'] = data['entryDate'].apply(lambda date: date[4:6])
    month_admission_binary, month_admission_label, _ = categorize_variable(data, 'month_admission')
    list_X.append(month_admission_binary)
    list_X_labels.append(month_admission_label)
    list_x_labels_predictor_title.append(['month_admission'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # month admission
    data['month_discharge'] = data['exitDate'].apply(lambda date: date[4:6])
    month_discharge_binary, month_discharge_label, _ = categorize_variable(data, 'month_discharge')
    list_X.append(month_discharge_binary)
    list_X_labels.append(month_discharge_label)
    list_x_labels_predictor_title.append(['month_discharge'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # day admission
    data['day_admission'] = data['entryDate'].apply(lambda date: datetime(int(date[:4]), int(date[4:6]), int(date[6:])).weekday())
    day_admission_binary, day_admission_label, _ = categorize_variable(data, 'day_admission')
    list_X.append(day_admission_binary)
    list_X_labels.append(day_admission_label)
    list_x_labels_predictor_title.append(['day_admission'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # day discharge
    data['day_discharge'] = data['exitDate'].apply(lambda date: datetime(int(date[:4]), int(date[4:6]), int(date[6:])).weekday())
    day_discharge_binary, day_discharge_label, _ = categorize_variable(data, 'day_discharge')
    list_X.append(day_discharge_binary)
    list_X_labels.append(day_discharge_label)
    list_x_labels_predictor_title.append(['day_discharge'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # has complex procedure
    #TODO find out how to figure out whether a procedure is complex

    # year of discharge
    #TODO probably not a good predictor, we any only predict always for cases from the same year

    # age of patient
    list_X.append(data['ageYears'].values.reshape(-1,1))
    list_X_labels.append(['age_years'])
    list_x_labels_predictor_title.append(['age_years'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # age bins of patient
    age_bin, age_bin_label = categorize_age(data['ageYears'].values, data['ageDays'].values)
    list_X.append(age_bin)
    list_X_labels.append(age_bin_label)
    list_x_labels_predictor_title.append(['age_bins'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # clinic id
    clinic_id_binary, clinic_id_label, _ = categorize_variable(data, 'Hauptkostenstelle')
    list_X.append(clinic_id_binary)
    list_X_labels.append(clinic_id_label)
    list_x_labels_predictor_title.append(['clinic_id'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # drg cost weight
    list_X.append(data['drgCostWeight'].values.reshape(-1,1))
    list_X_labels.append(['drg_cost_weight'])
    list_x_labels_predictor_title.append(['drg_cost_weight'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # effective cost weight
    list_X.append(data['effectiveCostWeight'].values.reshape(-1,1))
    list_X_labels.append(['effective_cost_weight'])
    list_x_labels_predictor_title.append(['effective_cost_weight'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # difference effective cost weight minus drg cost weight
    list_X.append(data['effectiveCostWeight'].values.reshape(-1,1) - data['drgCostWeight'].values.reshape(-1,1))
    list_X_labels.append(['effective_cost_weight_minus_drg_cost_weight'])
    list_x_labels_predictor_title.append(['effective_cost_weight_minus_drg_cost_weight'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # mdc
    mdc_binary, mdc_labels, _ = categorize_variable(data, 'mdc')
    list_X.append(mdc_binary)
    list_X_labels.append(mdc_labels)
    list_x_labels_predictor_title.append(['mdc'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # mdc partition
    mdc_partition_binary, mdc_partition_labels, _ = categorize_variable(data, 'mdcPartition')
    list_X.append(mdc_partition_binary)
    list_X_labels.append(mdc_partition_labels)
    list_x_labels_predictor_title.append(['mdc_partition'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # number of relevant diags
    list_X.append(data['NumDrgRelevantDiagnoses'].values.reshape(-1,1))
    list_X_labels.append(['num_drg_relevant_diagnoses'])
    list_x_labels_predictor_title.append(['num_drg_relevant_diagnoses'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # number of relevant chops
    list_X.append(data['NumDrgRelevantProcedures'].values.reshape(-1,1))
    list_X_labels.append(['num_drg_relevant_procedures'])
    list_x_labels_predictor_title.append(['num_drg_relevant_procedures'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # number of chops
    data['number_of_chops'] = data['procedures'].apply(lambda x: len(x))
    list_X.append(data['number_of_chops'].values.reshape(-1,1))
    list_X_labels.append(['number_of_chops'])
    list_x_labels_predictor_title.append(['number_of_chops'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # number of diags
    data['number_of_diags'] = data['secondaryDiagnoses'].apply(lambda x: len(x) + 1)
    list_X.append(data['number_of_diags'].values.reshape(-1,1))
    list_X_labels.append(['number_of_diags'])
    list_x_labels_predictor_title.append(['number_of_diags'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # number diagnoses with ccl > ß
    data['number_of_diags_ccl_greater_0'] = data['diagnosesExtendedInfo'].apply(__extract_number_of_ccl_greater_null)
    list_X.append(data['number_of_diags_ccl_greater_0'].values.reshape(-1,1))
    list_X_labels.append(['number_of_diags_ccl_greater_0'])
    list_x_labels_predictor_title.append(['number_of_diags_ccl_greater_0'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # pccl
    pccl_binary, pccl_label, _ = categorize_variable(data, 'pccl')
    list_X.append(pccl_binary)
    list_X_labels.append(pccl_label)
    list_x_labels_predictor_title.append(['pccl'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # raw pccl
    list_X.append(data['rawPccl'].values.reshape(-1,1))
    list_X_labels.append(['raw_pccl'])
    list_x_labels_predictor_title.append(['raw_pccl'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # duration of say
    list_X.append(data['durationOfStay'].values.reshape(-1,1))
    list_X_labels.append(['duration_of_stay'])
    list_x_labels_predictor_title.append(['duration_of_stay'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # gender
    gender_binary, gender_labels, _ = categorize_variable(data, 'gender')
    list_X.append(gender_binary)
    list_X_labels.append(gender_labels)
    list_x_labels_predictor_title.append(['gender'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # leave days
    list_X.append(data['leaveDays'].values.reshape(-1,1))
    list_X_labels.append(['leave_days'])
    list_x_labels_predictor_title.append(['leave_days'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # age flag
    list_X.append(data['ageFlag'].values.reshape(-1,1))
    list_X_labels.append(['age_flag'])
    list_x_labels_predictor_title.append(['age_flag'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # age flag
    list_X.append(data['admissionWeightFlag'].values.reshape(-1,1))
    list_X_labels.append(['admission_weight_flag'])
    list_x_labels_predictor_title.append(['admission_weight_flag'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # age flag
    list_X.append(data['genderFlag'].values.reshape(-1,1))
    list_X_labels.append(['gender_flag'])
    list_x_labels_predictor_title.append(['gender_flag'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # age flag
    list_X.append(data['grouperAdmissionCodeFlag'].values.reshape(-1,1))
    list_X_labels.append(['grouper_admission_code_flag'])
    list_x_labels_predictor_title.append(['grouper_admission_code_flag'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # age flag
    list_X.append(data['grouperDischargeCodeFlag'].values.reshape(-1,1))
    list_X_labels.append(['grouper_discharge_code_flag'])
    list_x_labels_predictor_title.append(['grouper_discharge_code_flag'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # age flag
    list_X.append(data['durationOfStayFlag'].values.reshape(-1,1))
    list_X_labels.append(['duration_of_stay_flag'])
    list_x_labels_predictor_title.append(['duration_of_stay_flag'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # age flag
    list_X.append(data['hoursMechanicalVentilationFlag'].values.reshape(-1,1))
    list_X_labels.append(['hours_mechanical_ventilation_flag'])
    list_x_labels_predictor_title.append(['hours_mechanical_ventilation_flag'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # age flag
    list_X.append(data['gestationAgeFlag'].values.reshape(-1,1))
    list_X_labels.append(['gestation_age_Flag'])
    list_x_labels_predictor_title.append(['gestation_age_Flag'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # duration of stay case type
    duration_of_stay_case_type_binary, duration_of_stay_case_type_label, _ = categorize_variable(data, 'durationOfStayCaseType')
    list_X.append(duration_of_stay_case_type_binary)
    list_X_labels.append(duration_of_stay_case_type_label)
    list_x_labels_predictor_title.append(['duration_of_stay_case_type'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # grouper status
    grouper_status_binary, grouper_status_label, _ = categorize_variable(data, 'grouperStatus')
    list_X.append(grouper_status_binary)
    list_X_labels.append(grouper_status_label)
    list_x_labels_predictor_title.append(['grouper_status'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # adrg
    data['adrg'] = data['drg'].apply(lambda x: x[:3])
    adrg_binary, adrg_labels, _ = categorize_variable(data, 'adrg')
    list_X.append(adrg_binary)
    list_X_labels.append(adrg_labels)
    list_x_labels_predictor_title.append(['adrg'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # supplement charges
    list_X.append(data['supplementCharges'].values.reshape(-1,1))
    list_X_labels.append(['supplement_charges'])
    list_x_labels_predictor_title.append(['supplement_charges'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # Aufenthalt Nach Austritt
    stay_after_discharge_binary, stay_after_discharge_label, _ = categorize_variable(data, 'AufenthaltNachAustritt')
    list_X.append(stay_after_discharge_binary)
    list_X_labels.append(stay_after_discharge_label)
    list_x_labels_predictor_title.append(['stay_after_discharge'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # Aufenthalts Klasse
    stay_class_binary, stay_class_label, _ = categorize_variable(data, 'AufenthaltsKlasse')
    list_X.append(stay_class_binary)
    list_X_labels.append(stay_class_label)
    list_x_labels_predictor_title.append(['stay_class'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # Aufenthaltsort Vor Dem Eintritt
    place_before_admission_binary, place_before_admission_label, _ = categorize_variable(data, 'AufenthaltsortVorDemEintritt')
    list_X.append(place_before_admission_binary)
    list_X_labels.append(place_before_admission_label)
    list_x_labels_predictor_title.append(['place_before_admission'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # Behandlung Nach Austritt
    treatment_after_discharge_binary, treatment_after_discharge_label, _ = categorize_variable(data, 'BehandlungNachAustritt')
    list_X.append(treatment_after_discharge_binary)
    list_X_labels.append(treatment_after_discharge_label)
    list_x_labels_predictor_title.append(['treatment_after_discharge'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # Eintrittsart
    type_of_admission_binary, type_of_admission_label, _ = categorize_variable(data, 'Eintrittsart')
    list_X.append(type_of_admission_binary)
    list_X_labels.append(type_of_admission_label)
    list_x_labels_predictor_title.append(['type_of_admission'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # Einweisende Instanz
    instructing_instance_binary, instructing_instance_label, _ = categorize_variable(data, 'EinweisendeInstanz')
    list_X.append(instructing_instance_binary)
    list_X_labels.append(instructing_instance_label)
    list_x_labels_predictor_title.append(['instructing_instance'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # Entscheid Fuer Austritt
    discharge_reason_binary, discharge_reason_label, _ = categorize_variable(data, 'EntscheidFuerAustritt')
    list_X.append(discharge_reason_binary)
    list_X_labels.append(discharge_reason_label)
    list_x_labels_predictor_title.append(['discharge_reason'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    # Hauptkostentraeger Fuer Grundversicherungsleistungen
    main_cost_unit_insurance_binary, main_cost_unit_insurance_label, _ = categorize_variable(data, 'HauptkostentraegerFuerGrundversicherungsleistungen')
    list_X.append(main_cost_unit_insurance_binary)
    list_X_labels.append(main_cost_unit_insurance_label)
    list_x_labels_predictor_title.append(['main_cost_unit_insurance'])
    logger.info(f'Extracted {list_x_labels_predictor_title[-1]}')

    return list_X, list_X_labels, list_x_labels_predictor_title

def __int_to_bool(row):
    if isinstance(row, int):
        return row > 0
    else:
        return False


def __extract_number_of_ccl_greater_null(row: dict):
    num_ccl_greater_null = 0
    for key in row.keys():
        if row[key]['ccl'] > 0:
            num_ccl_greater_null += 1
    return num_ccl_greater_null


def categorize_variable(data: pd.DataFrame, variable: str, encoder: object = None) -> (npt.ArrayLike, list, object):
    """ Categorize a variable in the DataFrame while training the encoder or using a given encoder.

    @param data: The DataFrame containing the variable which should be categorized.
    @param variable: The variable name which should be categorized.
    @param encoder: If given, the encoder is used to categorize.
    @return: (the categorized variable, the list of class labels, the encoder)
    """
    assert variable in data.columns, "Variable not contained in the given DataFrame."
    logger.info(f'Start categorizing variable {variable}.')
    input_is_set = isinstance(data[variable].values[0], set)
    if encoder is None:
        logger.info(f'Fitting a new encoder for variable {variable}.')
        if input_is_set:
            sorted_classes = None
            encoder = MultiLabelBinarizer(classes=sorted_classes).fit(data[variable].values.tolist())
        else:
            sorted_classes = np.sort(data[variable].unique())
            encoder = MultiLabelBinarizer(classes=sorted_classes).fit(data[variable].values.reshape((-1,1)))

    if input_is_set:
        encoded_variable = encoder.transform(data[variable].values.tolist())
    else:
        encoded_variable = encoder.transform(data[variable].values.reshape((-1, 1)))
    logger.info(f'Categorized variable {variable}. Shape of encoded variable is {encoded_variable.shape}')
    return encoded_variable, [f'{variable}_{x}' for x in encoder.classes_], encoder


def categorize_age(ages_years: npt.ArrayLike, ages_days: npt.ArrayLike):
    """ Compute age bins

    @param ages_years: The age in years.
    @param ages_days: The age in days.
    :return: (The age-categorized BFS data, the labels for the age bins)
    """
    agebins_labels = ['age_below_28_days', 'age_28_days_to_2_years', 'age_2_to_5_years', 'age_6_to_15_years', 'age_16_to_29_years', 'age_30_to_39_years', 'age_40_to_49_years', 'age_50_to_59_years', 'age_60_to_69_years', 'age_70_to_79_years', 'age_80_and_older']
    categories_age = np.zeros((len(ages_years), len(agebins_labels)))
    for i, age_year_days in enumerate(zip(ages_years, ages_days)):
        age_year = age_year_days[0]
        age_day = age_year_days[1]

        if age_year == 0 and 0 <= age_day < 28:
            categories_age[i, 0] = 1
        elif 28 <= age_day <= 365 or 1 <= age_year < 2:
            categories_age[i, 1] = 1
        elif 2 <= age_year <= 5:
            categories_age[i, 2] = 1
        elif 6 <= age_year <= 15:
            categories_age[i, 3] = 1
        elif 16 <= age_year <= 29:
            categories_age[i, 4] = 1
        elif 30 <= age_year <= 39:
            categories_age[i, 5] = 1
        elif 40 <= age_year <= 49:
            categories_age[i, 6] = 1
        elif 50 <= age_year <= 59:
            categories_age[i, 7] = 1
        elif 60 <= age_year <= 69:
            categories_age[i, 8] = 1
        elif 70 <= age_year <= 79:
            categories_age[i, 9] = 1
        else:
            categories_age[i, 10] = 1

    return categories_age, agebins_labels


def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))

