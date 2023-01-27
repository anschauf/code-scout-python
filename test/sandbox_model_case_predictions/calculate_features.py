import gc
import os.path
import sys
from os.path import join

from loguru import logger

from src import ROOT_DIR
from test.sandbox_model_case_predictions.data_handler import engineer_mind_bend_suggestions, load_data
from test.sandbox_model_case_predictions.utils import get_list_of_all_predictors, get_revised_case_ids

# -----------------------------------------------------------------------------
OVERWRITE_REVISED_CASE_IDs = False
OVERWRITE_FEATURE_FILES = True
# -----------------------------------------------------------------------------


def calculate_features():
    """Calculate the features and store the info regarding whether a case is reviewed or revised.
    Note: It's actually faster to read the columns / features in batches.
    """
    features_dir = join(ROOT_DIR, 'resources', 'features')
    revised_case_ids_filename = join(ROOT_DIR, 'resources', 'data', 'revised_case_ids.csv')

    # Read only the columns needed to list the reviewed and revised cases
    if OVERWRITE_REVISED_CASE_IDs:
        all_data = load_data(columns=['id', 'AnonymerVerbindungskode', 'ageYears', 'gender', 'durationOfStay', 'hospital', 'exitDate'])
        get_revised_case_ids(all_data, revised_case_ids_filename, overwrite=True)

    if OVERWRITE_FEATURE_FILES:
        # shutil.rmtree(features_dir, ignore_errors=True)

        column_batches = [
            # ['ageYears', 'ageDays', 'ErfassungDerAufwandpunkteFuerIMC', 'AufenthaltIntensivstation',
            #  'NEMSTotalAllerSchichten', 'durationOfStay', 'leaveDays', 'hospital', 'drgCostWeight',
            #  'effectiveCostWeight', 'NumDrgRelevantDiagnoses', 'NumDrgRelevantProcedures', 'rawPccl',
            #  'supplementCharges'],
            # ['gender', 'Hauptkostenstelle', 'mdc', 'mdcPartition', 'durationOfStayCaseType', 'AufenthaltNachAustritt',
            #  'AufenthaltsKlasse', 'Eintrittsart', 'EntscheidFuerAustritt', 'AufenthaltsortVorDemEintritt',
            #  'BehandlungNachAustritt', 'EinweisendeInstanz', 'HauptkostentraegerFuerGrundversicherungsleistungen',
            #  'grouperDischargeCode', 'grouperAdmissionCode'],
            # ['AufenthaltIntensivstation', 'NEMSTotalAllerSchichten', 'ErfassungDerAufwandpunkteFuerIMC',
            #  'IsCaseBelowPcclSplit', 'ageFlag', 'genderFlag', 'durationOfStayFlag', 'grouperAdmissionCodeFlag',
            #  'grouperDischargeCodeFlag', 'hoursMechanicalVentilationFlag', 'gestationAgeFlag', 'admissionWeightFlag',
            #  'effectiveCostWeight', 'drgCostWeight'],
            # ['primaryDiagnosis', 'secondaryDiagnoses', 'procedures', 'diagnosesExtendedInfo', 'proceduresExtendedInfo'],
            # ['hoursMechanicalVentilation', 'mdc', 'medications', 'entryDate', 'exitDate', 'pccl', 'rawPccl'],
            # ['VectorizedCodes'],
        ]

        for column_batch in column_batches:
            all_data = load_data(columns=column_batch)
            feature_filenames, _ = get_list_of_all_predictors(all_data, features_dir, overwrite=False)
            feature_names = sorted(list(feature_filenames.keys()))
            n_features = len(feature_names)
            logger.success(f'Created {n_features} features')

            # Delete the large DataFrame from memory and force collection
            del all_data
            gc.collect()

        # ---------------------------------------------------------------------
        # MindBend suggestions
        # ---------------------------------------------------------------------
        revised_case_info_df = get_revised_case_ids(None, revised_case_ids_filename, overwrite=False)

        mind_bend_features = engineer_mind_bend_suggestions(
            revised_case_info_df=revised_case_info_df,
            # files_path='s3://code-scout/mind_bend_output/revised_cases/',
            files_path=os.path.join(ROOT_DIR, 'resources', 'mind_bend_suggestions', 'ksw_2020'),
        )

        feature_filename = os.path.join(features_dir, f'mind_bend_ksw_2020.csv')
        logger.info(f'Storing MindBend features at {feature_filename} ...')
        mind_bend_features.to_csv(feature_filename, index=False)


if __name__ == '__main__':
    calculate_features()
    logger.success('done')
    sys.exit(0)
