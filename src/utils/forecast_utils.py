import numpy as np
import pandas as pd
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import os
import sys
from dotenv import load_dotenv
load_dotenv()
REPO_PATH= os.getenv('REPO_PATH')
sys.path.insert(0, rf'{REPO_PATH}src')

from utils.model_utils import build_rnn_model, RNNGenerator


# RNN forecast model
class ForecastModel:
    def __init__(
            self,
            model_name: str,
        ):

        self.model_name = model_name

        with open(
            f'model_archive/{self.model_name}/data_params.json',
            'r'
            ) as f:
            self.data_params = json.load(f)
        with open(
            f'model_archive/{self.model_name}/model_params.json',
            'r'
            ) as f:
            self.model_params = json.load(f)

        model_comp = model_name.split('_')
        self.future = model_comp[0]
        self.rnn_type = model_comp[1]

        self.gen = RNNGenerator(
            self.future,
            self.data_params['CV']
        )

        self.gen.preprocess_data(
            self.data_params['feature_columns'],
            self.data_params['target_column'],
            self.data_params['window_size'],
            test_size=self.data_params['test_size'],
            val_size=self.data_params['val_size'],
        )

        # Build the model
        self.model = build_rnn_model(
            self.rnn_type,
            self.model_params,
            (
                self.data_params['window_size'],
                len(self.data_params['feature_columns'])
            )
        )

        self.model.load_weights(
            f'model_archive/{self.model_name}/model_weights.h5'
        )

        self.test_predictions = self.model.predict(
            self.gen.test_generator
        ).flatten()

        self.test_targets = np.concatenate(
            [y for _, y in self.gen.test_generator]
        ).flatten()

    def describe(
            self,
            decimals: float = 4,
            print: bool = False
        ) -> None | dict[str, any]:

        metrics = calculate_metrics(
            self.test_targets,
            self.test_predictions,
            decimals=decimals
        )

        if print:
            print(f'Model: {self.model_name}')
            for name, metric in metrics.items():
                print(f'{name.upper()}: {metric}')
        else:
            return metrics


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
