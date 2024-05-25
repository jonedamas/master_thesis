import numpy as np
import pandas as pd
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error
from arch import arch_model

import os
import sys
from dotenv import load_dotenv
load_dotenv()
REPO_PATH= os.getenv('REPO_PATH')
sys.path.insert(0, rf'{REPO_PATH}src_HF')

from utils.model_utils import build_rnn_model, RNNGenerator


class ForecastGARCH:
    def __init__(self,
        model_name: str,
        model_params: dict[str, any],
        data_params: dict[str, any]
        ):

        self.model_name = model_name
        self.model_params = model_params
        self.data_params = data_params


    def describe(self):
        print(f'Model: {self.model_name}')
        print(f'MSE: {self.mse}')
        print(f'MAE: {self.mae}')


# RNN forecast model
class ForecastModel:
    def __init__(
            self,
            model_name: str,
        ):

        self.model_name = model_name

        with open(f'model_archive/{self.model_name}/data_params.json', 'r') as f:
            self.data_params = json.load(f)
        with open(f'model_archive/{self.model_name}/model_params.json', 'r') as f:
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

        self.mse = mean_squared_error(
            self.test_targets, self.test_predictions
        )
        self.mae = mean_absolute_error(
            self.test_targets, self.test_predictions
        )
        self.mda = mean_directional_accuracy(
            pd.Series(self.test_targets),
            pd.Series(self.test_predictions)
        )

    def describe(
            self,
            print: bool = False
        ) -> None | dict[str, any]:

        if print:
            print(f'Model: {self.model_name}')
            print(f'MSE: {self.mse}')
            print(f'MAE: {self.mae}')
            print(f'MDA: {self.mda}')
        else:
            return {
                'mse': self.mse,
                'mae': self.mae,
                'mda': self.mda
            }


def mean_directional_accuracy(
        y_true: pd.Series,
        y_pred: pd.Series
    ) -> float:
    """
    Calculate the directional accuracy of a forecast model

    Parameters
    ----------
    y_true : pd.Series
        The true target values
    y_pred : pd.Series
        The predicted target values

    Returns
    -------
    float
        The accuracy of the model in predicting the direction of the target
    """
    true_change = np.sign(y_true.diff().fillna(0))
    pred_change = np.sign((y_pred - y_true.shift()).fillna(0))

    accuracy = (true_change == pred_change).sum() / len(true_change)

    return accuracy
