import numpy as np
import pandas as pd
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
            model_params: dict[str, any],
            data_params: dict[str, any],
        ):
        self.model_name = model_name
        self.model_params = model_params
        self.data_params = data_params

        model_comp = model_name.split('_')
        self.future = model_comp[0]
        self.topic = model_comp[1]
        self.rnn_type = model_comp[2]

        self.gen = RNNGenerator(self.future, self.topic)

        self.gen.preprocess_data(
            self.data_params['feature_columns'],
            self.data_params['target_column'],
            self.data_params['window_size'],
            test_size=self.data_params['test_size'],
            val_size=self.data_params['val_size'],
            CV=self.data_params['CV']
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

        self.model.load_weights(f'model_archive/{self.model_name}.h5')


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

    def describe(self):
        print(f'Model: {self.model_name}')
        print(f'MSE: {self.mse}')
        print(f'MAE: {self.mae}')
