from typing import List

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score

from src.schema import prob_most_likely_code_col, suggested_code_rankings_split_col, case_id_col
from src.service.bfs_cases_db_service import get_patient_case_for_aimedic_ids_df
from src.service.database import Database


def load_data(meta_data: pd.DataFrame, load_diagnoses=False, load_procedures=False) -> pd.DataFrame:
    """ Load all cases given in the meta data containin a column 'aimedic_id'.

    @param meta_data: Meta data containing at least the column 'aimedic_id'.
    @param load_diagnoses: Whether to load the diagnoses.
    @param load_procedures: Whether to load the procedures.
    @return: Patient data, consisting of socio-demographics and revisions, merged with meta data.
    """
    logger.info(f'Start loading {meta_data.shape[0]} patient cases from DB ...')
    with Database() as db:
        data = get_patient_case_for_aimedic_ids_df(meta_data['aimedic_id'].values.tolist(), db.session, load_diagnoses=load_diagnoses, load_procedures=load_procedures)
        logger.info(f'... Loading of {data.shape[0]} patient cases from DB finished.')
        return pd.merge(data, meta_data, how='outer', on='aimedic_id')


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

