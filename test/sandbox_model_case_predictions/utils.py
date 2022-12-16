import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MultiLabelBinarizer


def get_list_of_all_predictors(data):
    # define predictors from data
    list_X = list()
    list_X_labels = list()

    # IMC effort points
    list_X.append(data['ErfassungDerAufwandpunkteFuerIMC'].values.reshape(-1, 1))
    list_X_labels.append('erfassung_der_aufwandpunkte_fuer_IMC')

    # Ventilation hours
    list_X.append(data['hoursMechanicalVentilation'].values.reshape(-1, 1))
    list_X_labels.append('hours_mechanical_ventilation')

    # Ventilation hours boolean
    ventilation_hours_boolean = np.asarray([int(x) for x in (data['hoursMechanicalVentilation'].values > 0)]).reshape(-1, 1)
    list_X.append(ventilation_hours_boolean)
    list_X_labels.append('hours_mechanical_ventilation_boolean')

    # Ventilation hours bins
    # left out, because not much data for ventilation hours

    # Ventilation hours (as a boolean) multiplied by ${DRG does not start with “A”}
    drg_starts_with_a = np.asarray([int(x) for x in data['drg'].apply(lambda x: True if (x.startswith('A')) else False).values]).reshape(-1, 1)
    list_X.append(ventilation_hours_boolean * drg_starts_with_a)
    list_X_labels.append('ventilation_hours_boolean_multiplied_with_DRG_starts_with_A')

    # Notfall/ Emergency boolean
    emergency = np.asarray([1 if (x=='1') else 0 for x in data['Eintrittsart'].values])
    list_X.append(emergency)
    list_X_labels.append('emergency_boolean')

    # Admission type
    admission_type, admisstion_type_label, _ = categorize_variable(data, 'Eintrittsart')
    list_X.append(admission_type)
    list_X_labels.append(admisstion_type_label)

    # discharge type
    discharge_type, discharge_type_label, _ = categorize_variable(data, 'EntscheidFuerAustritt')
    list_X.append(discharge_type)
    list_X_labels.append(discharge_type_label)

    # hours in ICU
    list_X.append(data['AufenthaltIntensivstation'].values.reshape(-1,1))
    list_X_labels.append('aufenthalt_intensivstation')

    # hours in ICU boolean
    hours_in_icu_boolean = data['AufenthaltIntensivstation'].values.reshape(-1,1) > 0
    list_X.append(hours_in_icu_boolean)
    list_X_labels.append('aufenthalt_intensivstation_boolean')

    # Does ADRG has PCCL-split boolean
    #TODO need infos

    # CCL sensitivity
    #TODO ask paolo for code

    # NEMS boolean
    nems_boolean = data['NEMSTotalAllerSchichten'].values.reshape(-1,1) > 0
    list_X.append(nems_boolean)
    list_X_labels.append('nems_total_aller_schichten')

    # IMC effort points boolean
    #TODO check if it has a true boolean with all data
    imc_effort_points_boolean = data['ErfassungDerAufwandpunkteFuerIMC'].values.reshape(-1,1) > 0
    list_X.append(imc_effort_points_boolean)
    list_X_labels.append('erfassung_der_aufwandpunkte_fuer_imc')

    # Medication ATC-code
    data['medications_atc'] = data['medications'].apply(lambda all_meds: set([x.split(':')[0] for x in all_meds]))
    medication_atc_codes_binary, medication_atc_codes_labels, _ = categorize_variable(data, 'medications_atc')
    list_X.append(medication_atc_codes_binary)
    list_X_labels.append(medication_atc_codes_labels)




    return list_X, list_X_labels


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