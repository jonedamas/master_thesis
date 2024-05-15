import numpy as np
import pandas as pd
from functools import partial
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Input, Dense, Dropout, BatchNormalization, GaussianNoise
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

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


class RNNGenerator:
    def __init__(self, future: str, topic: str):
        self.future = future
        self.topic = topic

        self.train_generator = None
        self.val_generator = None
        self.test_generator = None

        self.file_path = os.path.join(
            REPO_PATH,
            'data',
            'prepared_data',
            f"{future}_{topic}_5min_resampled.csv"
        )

        self.df = pd.read_csv(self.file_path, index_col='date', parse_dates=True)

    def __repr__(self) -> str:
        return f"RNNGenerator(future={self.future}, topic={self.topic})"


    def preprocess_data(
            self,
            feature_columns: list[str],
            target_column: str,
            window_size: int,
            test_size: float = 0.2,
            val_size: float = 0.2,
            batch_size: int = 32,
            scaler_type: str = 'StandardScaler'
        ) -> None:
        """
        Preprocess the data for LSTM-like models.

        Parameters:
        - df: DataFrame containing the dataset.
        - feature_columns: List of columns to be used as features.
        - target_column: Column to be used as target.
        - window_size: Number of past time steps to use as input features.
        - train_split: Fraction of the data to be used for training (default is 0.8).

        Returns:
        - train_generator: TimeseriesGenerator for training data.
        - val_generator: TimeseriesGenerator for validation data.
        - test_generator: TimeseriesGenerator for test data.
        """
        # Select features and target
        X: pd.Series = self.df[feature_columns]
        y: pd.Series = self.df[target_column]

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, shuffle=False
        )
        # scale data
        if scaler_type == 'StandardScaler':
            self.scaler = StandardScaler()
        elif scaler_type == 'MinMaxScaler':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError('Unsupported scaler type.')

        X_train: np.array = self.scaler.fit_transform(X_train)
        X_val: np.array = self.scaler.transform(X_val)
        X_test: np.array = self.scaler.transform(X_test)

        # Create sequences of window_size with TimeseriesGenerator
        self.train_generator = TimeseriesGenerator(
            X_train, y_train, length=window_size, batch_size=batch_size
        )
        self.val_generator = TimeseriesGenerator(
            X_val, y_val, length=window_size, batch_size=batch_size
        )
        self.test_generator = TimeseriesGenerator(
            X_test, y_test, length=window_size, batch_size=batch_size
        )


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


class ForecastModel:
    def __init__(
            self,
            model_name: str,
            model_params: dict[str, any],
            data_params: dict[str, any]
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
            val_size=self.data_params['val_size']
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
