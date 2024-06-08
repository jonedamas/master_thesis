import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna
import sqlite3


def mean_directional_accuracy(
        y_true,
        y_pred
    ) -> float:
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
        y_test,
        y_pred,
        decimals: int = 4
    ) -> dict[str, float]:
    """
    Calculate the mean squared error, mean absolute error, and mean directional accuracy.
    """

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mda = mean_directional_accuracy(y_test, y_pred)
    rmse = np.sqrt(mse)

    metric_dict = {
        'mse': round(mse, decimals),
        'mae': round(mae, decimals),
        'rmse': round(rmse, decimals),
        'mda': round(mda, decimals)
    }

    return metric_dict


def create_importance_df(
        file_loc: str
    ) -> pd.DataFrame:

    def get_importance(
            study_name: str,
            loc: str
        ) -> pd.Series | None:
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
