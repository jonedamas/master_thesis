import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import skew, kurtosis
import optuna
import sqlite3

from typing import Dict, Union


def mean_directional_accuracy(y_true: any, y_pred: any) -> float:
    """
    Calculate the directional accuracy of a forecast model

    Parameters
    ----------
    y_true
        The true target values
    y_pred
        The predicted target values

    Returns
    -------
    float
        The accuracy of the model in predicting the direction of the target
    """
    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true)

    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred)

    # y_true, y_pred = y_true.align(y_pred)

    true_change = np.sign(y_true.diff().fillna(0))
    pred_change = np.sign((y_pred - y_true.shift()).fillna(0))

    accuracy = (true_change == pred_change).sum() / len(true_change)

    return accuracy


def calculate_metrics(
        y_test: any,
        y_pred: any,
        decimals: int = 4
    ) -> Dict[str, float]:
    """
    Calculate the mean squared error, mean absolute error, and mean directional accuracy.

    Parameters
    ----------
    y_test
        The true target values
    y_pred
        The predicted target values
    decimals
        The number of decimal places to round the metrics to

    Returns
    -------
    dict[str, float]
        A dictionary of the calculated metrics
    """
    threshold = 1e-6
    mask = y_test >= threshold
    y_test_filtered = y_test[mask]
    y_pred_filtered = y_pred[mask]

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test_filtered, y_pred_filtered)
    mda = mean_directional_accuracy(y_test, y_pred)
    rmse = np.sqrt(mse)

    metric_dict = {
        'mse': round(mse, decimals),
        'mae': round(mae, decimals),
        'rmse': round(rmse, decimals),
        'mda': round(mda, decimals),
        'mape': round(mape, decimals)
    }

    return metric_dict


def create_importance_df(file_loc: str) -> pd.DataFrame:
    """
    Create a DataFrame of the importance of each parameter in each study

    Parameters
    ----------
    file_loc
        The location of the SQLite database file

    Returns
    -------
    pd.DataFrame
        A DataFrame of the importance of each parameter in each study
    """

    def get_importance(study_name: str,loc: str) -> Union[pd.Series, None]:
        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=f'sqlite:///{loc}'
            )

            importance = pd.Series(
                optuna.importance.get_param_importances(study)
            )
            return importance

        except:
            return None

    con = sqlite3.connect(file_loc)
    cur = con.cursor()

    table_list = [a[0] for a in cur.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    ]

    table_dict = dict()

    for table in table_list:
        param_df = pd.read_sql_query(
            f"SELECT * from {table}",
            con
        )
        table_dict[table] = param_df

    con.close()

    params_dict = {}

    for study_name in table_dict['studies']['study_name']:
        importance = get_importance(study_name, file_loc)
        std_name = ' '.join(study_name.split('_')[0:3])
        if importance is not None:
            importance['model_type'] = study_name.split('_')[1]

            params_dict[std_name] = importance

    params_df = pd.DataFrame(params_dict).T

    return params_df


def describe_df(input_series: pd.Series) -> pd.DataFrame:
    """
    Create a DataFrame of descriptive statistics for a series

    Parameters
    ----------
    input_series
        The series to describe

    Returns
    -------
    pd.DataFrame
        A DataFrame of the descriptive statistics
    """
    ddf = pd.DataFrame(input_series.describe()).T
    ddf['skew'] = skew(input_series.dropna())
    ddf['kurtosis'] = kurtosis(input_series.dropna())

    return ddf
