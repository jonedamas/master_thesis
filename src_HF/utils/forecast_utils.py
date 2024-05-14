import numpy as np
import pandas as pd
from functools import partial
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Input, Dense, Dropout, BatchNormalization, GaussianNoise
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from hpbandster.core.worker import Worker
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.configuration_space import ConfigurationSpace
from hpbandster.optimizers import BOHB as BOHB_Optimizer
import logging
from hpbandster.core.result import json_result_logger

import optuna
import os
from dotenv import load_dotenv
load_dotenv()
REPO_PATH= os.getenv('REPO_PATH')


rnn_layers: dict[str, any] = {
    'LSTM': LSTM,
    'BiLSTM': lambda units, **kwargs: Bidirectional(LSTM(units, **kwargs)),
    'GRU': GRU,
    'BiGRU': lambda units, **kwargs: Bidirectional(GRU(units, **kwargs))
}


def load_prepared_data(
        future: str,
        topic: str,
    ) -> pd.DataFrame:
    """
    Load data for specified future and topic with a given resample window size.

    Parameters:
    - future: str, one of the futures in the FUTURES list.
    - topic: str, one of the topics in the TOPICS list.
    - resample_window: str, resample window size, default is '5min'.

    Returns:
    - DataFrame of the loaded data.
    """

    file_name = f"{future}_{topic}_5min_resampled.csv"
    file_path = os.path.join(REPO_PATH, 'data', 'prepared_data', file_name)

    df = pd.read_csv(file_path, index_col='date', parse_dates=True)
    return df


def preprocess_data(
        df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        window_size: int,
        test_size: float = 0.2,
        val_size: float = 0.2,
        batch_size: int = 32
    ) -> tuple[TimeseriesGenerator]:
    """
    Preprocess the data for LSTM-like models.

    Parameters:
    - df: DataFrame containing the dataset.
    - feature_columns: List of columns to be used as features.
    - target_column: Column to be used as target.
    - window_size: Number of past time steps to use as input features.
    - train_split: Fraction of the data to be used for training (default is 0.8).

    Returns:
    - X_train, X_test, y_train, y_test: Training and testing data split.
    """
    # Select features and target
    X: pd.Series = df[feature_columns]
    y: pd.Series = df[target_column]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, shuffle=False
    )
    # scale data
    scaler = StandardScaler()
    X_train: np.array = scaler.fit_transform(X_train)
    X_val: np.array = scaler.transform(X_val)
    X_test: np.array = scaler.transform(X_test)

    # Create sequences of window_size with TimeseriesGenerator
    train_generator = TimeseriesGenerator(
        X_train, y_train, length=window_size, batch_size=batch_size
    )
    val_generator = TimeseriesGenerator(
        X_val, y_val, length=window_size, batch_size=batch_size
    )
    test_generator = TimeseriesGenerator(
        X_test, y_test, length=window_size, batch_size=batch_size
    )

    return train_generator, val_generator, test_generator



def optimize_hyperparameters(
        train_generator: TimeseriesGenerator,
        val_generator: TimeseriesGenerator,
        trial_config: dict[str, any],
        feature_columns: list[str],
        rnn_type: str = 'LSTM',
        window_size: int = 30,
        n_trials: int = 50,
        n_jobs: int = -1,
    ) -> dict[str, any]:
    """
    Optimize RNN model hyperparameters using Optuna for a given RNN type.

    Parameters
    ----------
    X_train : np.array
        Training feature data shaped (n_samples, timesteps, features).
    y_train : np.array
        Training target data shaped (n_samples,).
    rnn_type : str
        Type of RNN to use. Options are 'LSTM', 'BiLSTM', 'GRU', 'BiGRU'.
    n_trials : int
        Number of optimization trials.
    n_jobs : int
        The number of parallel jobs to run for optimization. If -1, use all available cores.

    Returns
    -------
    dict
        Best hyperparameters found during optimization.
    """

    def objective(trial, config: dict[str, any]):
        # Model configuration based on trial suggestions
        units_first_layer = trial.suggest_categorical(
            'units_first_layer', config['units_first_layer']
        )
        units_second_layer = trial.suggest_categorical(
            'units_second_layer', config['units_second_layer']
        )
        dropout_rate_first = trial.suggest_float(
            'dropout_rate_first', *config['dropout_rate_first']
        )
        dropout_rate_second = trial.suggest_float(
            'dropout_rate_second', *config['dropout_rate_second']
        )
        l2_strength = trial.suggest_float(
            'l2_strength', *config['l2_strength'], log=True
        )
        learning_rate = trial.suggest_float(
            'learning_rate', *config['learning_rate'], log=True
        )
        batch_size = trial.suggest_categorical(
            'batch_size', config['batch_size']
        )
        noise_std = trial.suggest_float(
            'noise_std', *config['noise_std']
        )

        # Selecting the model type
        if rnn_type in rnn_layers:
            rnn_layer = rnn_layers[rnn_type]
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")

        # Building the model
        model = Sequential([
            Input(shape=(window_size, len(feature_columns))),
            GaussianNoise(noise_std),
            rnn_layer(
                units_first_layer,
                return_sequences=True,
                kernel_regularizer=l2(l2_strength)
            ),
            Dropout(dropout_rate_first),
            BatchNormalization(),
            rnn_layer(
                units_second_layer,
                return_sequences=False,
                kernel_regularizer=l2(l2_strength)
            ),
            Dropout(dropout_rate_second),
            BatchNormalization(),
            Dense(1, activation='linear')
        ])

        # Compile the model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )

        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # Train the model
        history = model.fit(
            train_generator,
            epochs=50,
            batch_size=int(batch_size),
            validation_data=val_generator,
            callbacks=[early_stopping],
            verbose=0
        )
        return np.min(history.history['val_loss'])

    study = optuna.create_study(direction="minimize")

    study.optimize(
        partial(objective, config=trial_config),
        n_trials=n_trials,
        n_jobs=n_jobs
    )

    return study.best_params


def build_rnn_model(
        rnn_type: str,
        best_params: dict[str, any],
        input_shape: tuple[int, int]
    ) -> Sequential:
    """
    Build RNN model based on the type and provided hyperparameters.

    Parameters:
    - rnn_type : str, type of RNN to build ('LSTM', 'BiLSTM', 'GRU', 'BiGRU')
    - best_params : dict, dictionary containing optimized hyperparameters
    - input_shape : tuple, shape of the input data

    Returns:
    - model: Compiled TensorFlow/Keras model.
    """
    # Selecting the model type
    if rnn_type in rnn_layers:
        rnn_layer = rnn_layers[rnn_type]
    else:
        raise ValueError(f"Unsupported RNN type: {rnn_type}")

    model = Sequential([
        Input(shape=input_shape),
        GaussianNoise(best_params['noise_std']),
        rnn_layer(best_params['units_first_layer'], return_sequences=True, kernel_regularizer=l2(best_params['l2_strength'])),
        Dropout(best_params['dropout_rate_first']),
        BatchNormalization(),
        rnn_layer(best_params['units_second_layer'], return_sequences=False, kernel_regularizer=l2(best_params['l2_strength'])),
        Dropout(best_params['dropout_rate_second']),
        BatchNormalization(),
        Dense(1, activation='linear')
    ])

    # Compile the model
    optimizer = Adam(learning_rate=best_params['learning_rate'])
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])

    return model


class RNNWorker(Worker):
    """
    A worker class for the BOHB optimizer to train and evaluate RNN models.

    Attributes:
        X_train (np.array): Training features data.
        y_train (np.array): Training target data.
        feature_columns (list): List of feature column names.
        rnn_type (str): Type of the RNN model ('LSTM', 'BiLSTM', 'GRU', 'BiGRU').
        window_size (int): Number of past time steps used as input features.
    """
    def __init__(self, X_train, y_train, feature_columns, rnn_type, window_size, **kwargs):
        super().__init__(**kwargs)
        self.X_train = X_train
        self.y_train = y_train
        self.feature_columns = feature_columns
        self.rnn_type = rnn_type
        self.window_size = window_size

    def compute(self, config, budget, **kwargs):
        """
        Trains and evaluates the RNN model using configurations and budget provided by BOHB.

        Parameters:
            config (dict): Configuration parameters for the model provided by BOHB.
            budget (float): Fractional budget to use for the training epochs.

        Returns:
            dict: Contains the loss and optional additional info about the model.
        """
        # Extract parameters from config
        units_first_layer = config['units_first_layer']
        units_second_layer = config['units_second_layer']
        dropout_rate_first = config['dropout_rate_first']
        dropout_rate_second = config['dropout_rate_second']
        l2_strength = config['l2_strength']
        learning_rate = config['learning_rate']
        noise_std = config['noise_std']

        model = build_rnn_model(self.rnn_type, config, (self.window_size, len(self.feature_columns)))
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])

        # Early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Fit the model
        model.fit(self.X_train, self.y_train, epochs=int(budget), batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)

        # Evaluate the model
        _, val_mae = model.evaluate(self.X_train, self.y_train, verbose=0)
        return ({
            'loss': val_mae,  # This is the metric to minimize
            'info': {}  # Additional optional information
        })



    @staticmethod
    def get_configspace():
        config_space = ConfigurationSpace()

        # Explicit integer casting to avoid any possible type confusion
        units_first_layer = UniformIntegerHyperparameter(
            "units_first_layer", lower=16, upper=128, default_value=int(64))
        units_second_layer = UniformIntegerHyperparameter(
            "units_second_layer", lower=16, upper=96, default_value=int(32))
        dropout_rate_first = UniformFloatHyperparameter(
            "dropout_rate_first", lower=0.1, upper=0.5, default_value=0.3)  # Correct as float
        dropout_rate_second = UniformFloatHyperparameter(
            "dropout_rate_second", lower=0.1, upper=0.5, default_value=0.3)  # Correct as float
        l2_strength = UniformFloatHyperparameter(
            "l2_strength", lower=1e-5, upper=1e-3, default_value=0.0001, log=True)  # Explicit float
        learning_rate = UniformFloatHyperparameter(
            "learning_rate", lower=1e-5, upper=1e-2, default_value=0.001, log=True)  # Explicit float
        batch_size = UniformIntegerHyperparameter(
            "batch_size", lower=16, upper=64, default_value=int(32))  # Explicit integer casting

        noise_std = UniformFloatHyperparameter(
            "noise_std", lower=0.01, upper=0.1, default_value=0.05)  # Correct as float

        config_space.add_hyperparameters([
            units_first_layer,
            units_second_layer,
            dropout_rate_first,
            dropout_rate_second,
            l2_strength,
            learning_rate,
            batch_size,
            noise_std
        ])
        return config_space




def optimize_hyperparameters_bohb(X_train, y_train, feature_columns, rnn_type, window_size, min_budget, max_budget, n_iterations):
    """
    Utilizes BOHB to optimize hyperparameters for specified RNN model type.

    Parameters:
        X_train (np.array): Training feature data.
        y_train (np.array): Training target data.
        feature_columns (list): List of feature column names used in the model.
        rnn_type (str): Type of RNN model to optimize ('LSTM', 'BiLSTM', 'GRU', 'BiGRU').
        window_size (int): Number of time steps in input sequences.
        min_budget (int): Minimum number of epochs for model training.
        max_budget (int): Maximum number of epochs for model training.
        n_iterations (int): Number of BOHB iterations to perform.

    Returns:
        dict: Best configuration found by the BOHB optimizer.
    """
    worker = RNNWorker(X_train, y_train, feature_columns, rnn_type, window_size, run_id='0')
    result_logger = json_result_logger(directory='.', overwrite=True)
    bohb = BOHB_Optimizer(configspace=worker.get_configspace(),
                          run_id='0',
                          min_budget=min_budget,
                          max_budget=max_budget,
                          result_logger=result_logger)

    res = bohb.run(n_iterations=n_iterations)
    bohb.shutdown(shutdown_workers=True)
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    best_config = id2config[incumbent]['config']

    return best_config
