from typing import List

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.preprocessing import MultiLabelBinarizer

from src.schema import prob_most_likely_code_col, suggested_code_rankings_split_col, case_id_col
from src.service.bfs_cases_db_service import get_patient_case_for_aimedic_ids_df
from src.service.database import Database


def load_data(meta_data: pd.DataFrame, load_diagnoses=False, load_procedures=False, only_revised_cases=False) -> pd.DataFrame:
    """ Load all cases given in the meta data containin a column 'aimedic_id'.

    @param meta_data: Meta data containing at least the column 'aimedic_id'.
    @param load_diagnoses: Whether to load the diagnoses.
    @param load_procedures: Whether to load the procedures.
    @param only_revised_cases: Whether to only load revised cases.
    @return: Patient data, consisting of socio-demographics and revisions, merged with meta data.
    """
    logger.info(f'Start loading {meta_data.shape[0]} patient cases from DB ...')

    if only_revised_cases:
        meta_data_revised = meta_data[meta_data['y_label_is_revised_case'] == 1]
        logger.info(f'Filtering {meta_data.shape[0] - meta_data_revised.shape[0]} which are not revised.')
        meta_data = meta_data_revised

    with Database() as db:
        data = get_patient_case_for_aimedic_ids_df(meta_data['aimedic_id'].values.tolist(), db.session, load_diagnoses=load_diagnoses, load_procedures=load_procedures)
        logger.info(f'... Loading of {data.shape[0]} patient cases from DB finished.')
        return pd.merge(data, meta_data, how='right', on='aimedic_id')


def train_lr_model(X: npt.ArrayLike, y: npt.ArrayLike, **kwargs):
    """ Train a Logistic Regression.

    @param X: Predictors matrix.
    @param y: Labels vector.
    @param fit_intercept: Whether to fit an intercept.
    @return: The trained model.
    """
    assert len(X) == len(y), "Predictors and labels need to have same amount of samples."
    logger.info('Training logistic regression')
    model = LogisticRegression(**kwargs)
    model = model.fit(X, y)
    return model


def predict_proba(model, predictors: npt.ArrayLike) -> npt.ArrayLike:
    """ Predict probabilities given a model and predictors.

    @param model: The model to predict with.
    @param predictors: Predictors matrix.
    @return: Vector containing probabilities for label==1 to be true.
    """
    return model.predict_proba(predictors)[:, 1]


def write_model_coefs_to_file(model, filename: str, predictor_labels: List[str]):
    """ Write model coefficients to file.

    @param model: The model containing the coefficients of interest.
    @param filename: The filename where to write the result to.
    @param predictor_labels: List of labels of the coefficients.
    """
    logger.info("Write Logistic Regression coefficients to file.")
    argsort_features_random_forest = np.argsort(np.abs(model.coef_.flatten()))[::-1]

    pd.DataFrame({
        'feature': np.asarray(predictor_labels)[argsort_features_random_forest],
        'coef': model.coef_.flatten()[argsort_features_random_forest]
    }).sort_values(by='coef', ascending=False).to_csv(filename, index=False)


def write_evaluation_metrics_to_file(y_train: npt.ArrayLike, probas_train: npt.ArrayLike, y_test: npt.ArrayLike, probas_test: npt.ArrayLike, filename: str, threshold: float=0.5):
    """ Evaluate model predictions using F1, recall and precision.

    @param y_train: The training data labels.
    @param probas_train: The training predictions.
    @param y_test: The test data labels.
    @param probas_test: The test predictions.
    @param filename: The filename where to write the results to.
    @param threshold: A threshold where to cut the probabilities.
    """
    assert 0 < threshold < 1, "Threshold must be between 0 and 1."
    predictions_train = [1 if (x > threshold) else 0 for x in probas_train]
    f1_training = f1_score(y_true=y_train, y_pred=predictions_train)
    recall_training = recall_score(y_true=y_train, y_pred=predictions_train)
    precision_training = precision_score(y_true=y_train, y_pred=predictions_train)

    predictions_test = [1 if (x > threshold) else 0 for x in probas_test]
    f1_test = f1_score(y_true=y_test, y_pred=predictions_test)
    recall_test = recall_score(y_true=y_test, y_pred=predictions_test)
    precision_test = precision_score(y_true=y_test, y_pred=predictions_test)

    pd.DataFrame({
        'f1': [f1_training, f1_test],
        'recall': [recall_training, recall_test],
        'precision': [precision_training, precision_test]
    }, index=['training', 'test']).to_csv(filename)


def extract_case_ranking_performance_app(data: pd.DataFrame, probas: npt.ArrayLike, filename: str):
    """ Given the patient data and revise-probability this function stores the results in a format for the case ranking app.

    @param data: DataFrame containing at least the aimedic_id.
    @param probas: Probabilities for the case to be revisable.
    @param filename: The location to store the file.
    """
    data.rename(columns={'aimedic_id': case_id_col}, inplace=True)
    data[prob_most_likely_code_col] = probas
    data[suggested_code_rankings_split_col] = ['']*len(probas)
    data[[case_id_col, suggested_code_rankings_split_col, 'UpcodingConfidenceScore']].to_csv(filename, sep=';', index=False)


def categorize_variable(data: pd.DataFrame, variable: str, encoder: object = None) -> (npt.ArrayLike, list, object):
    """ Categorize a variable in the DataFrame while training the encoder or using a given encoder.

    @param data: The DataFrame containing the variable which should be categorized.
    @param variable: The variable name which should be categorized.
    @param encoder: If given, the encoder is used to categorize.
    @return: (the categorized variable, the list of class labels, the encoder)
    """
    assert variable in data.columns, "Variable not contained in the given DataFrame."
    logger.info(f'Start categorizing variable {variable}.')
    if encoder is None:
        logger.info(f'Fitting a new encoder for variable {variable}.')
        encoder = MultiLabelBinarizer(classes=np.sort(data[variable].unique())).fit(data[variable].values.reshape((-1,1)))
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