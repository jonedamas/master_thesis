import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


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
