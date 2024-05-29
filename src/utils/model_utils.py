import numpy as np
import pandas as pd
import json
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from functools import partial
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit

from keras.models import Sequential
from keras.layers import LSTM, GRU, Bidirectional, Input, Dense, GaussianNoise
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import TimeseriesGenerator

import optuna
import os
import sys
from dotenv import load_dotenv
load_dotenv()
REPO_PATH= os.getenv('REPO_PATH')

sys.path.insert(0, rf'{REPO_PATH}src')
from utils.main_utils import load_processed


RNN_LAYERS: dict[str, any] = {
    'LSTM': LSTM,
    'BiLSTM': lambda units, **kwargs: Bidirectional(LSTM(units, **kwargs)),
    'GRU': GRU,
    'BiGRU': lambda units, **kwargs: Bidirectional(GRU(units, **kwargs))
}

SCALERS: dict[str, any] = {
    'StandardScaler': StandardScaler,
    'MinMaxScaler': MinMaxScaler,
    'RobustScaler': RobustScaler
}


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
    if rnn_type in RNN_LAYERS:
        rnn_layer = RNN_LAYERS[rnn_type]
    else:
        raise ValueError(f"Unsupported RNN type: {rnn_type}")

    model = Sequential([
        Input(shape=input_shape),
        GaussianNoise(best_params['noise_std']),
        rnn_layer(
            best_params['units_first_layer'],
            return_sequences=True,
            kernel_regularizer=l2(best_params['l2_strength'])
        ),
        rnn_layer(
            best_params['units_second_layer'],
            return_sequences=False,
            kernel_regularizer=l2(best_params['l2_strength'])
        ),
        Dense(1, activation='linear')
    ])

    # Compile the model
    optimizer = Adam(learning_rate=best_params['learning_rate'])

    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )

    return model


class RNNGenerator:
    def __init__(
            self,
            future: str,
            CV: bool = False,
            CVfolds: int = 5
        ):
        """
        Initialize the RNNGenerator class.

        Parameters
        ----------
        future : str
            The futures data to use.
        CV : bool
            Whether to use cross-validation or not.
        CVfolds : int
            Number of cross-validation folds.
        """
        self.future = future
        self.CV = CV

        # Loading preprocessed data
        self.df = load_processed(self.future)[future]
        self.tscv = TimeSeriesSplit(n_splits=CVfolds)

        self.test_dates: None | pd.DatetimeIndex = None
        self.train_dates: None | pd.DatetimeIndex = None

        self.train_generators: list[any] = list()
        self.val_generators: list[any] = list()
        self.test_generator: None | any = None

    def __repr__(self) -> str:
        return f"RNNGenerator(future={self.future}), CV={self.CV})"

    def preprocess_data(
            self,
            feature_columns: list[str],
            target_column: str,
            window_size: int,
            test_size: float = 0.2,
            val_size: float = 0.2,
            batch_size: int = 32,
            scaler_type: str = 'RobustScaler'
        ) -> None:
        """
        Preprocess the data for LSTM-like models.

        Parameters
        ----------
        feature_columns : list
            List of feature columns to use.
        target_column : str
            Target column to predict.
        window_size : int
            Number of time steps to use for prediction.
        test_size : float
            Fraction of data to use for testing.
        val_size : float
            Fraction of data to use for validation.
        batch_size : int
            Batch size for training.
        scaler_type : str
            Type of scaler to use.
        """
        # Select features and target
        X: pd.Series = self.df[feature_columns]
        y: pd.Series = self.df[target_column]

        # scale data
        if scaler_type in SCALERS:
            self.scaler = SCALERS[scaler_type]()
        else:
            raise ValueError('Unsupported scaler type.')

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        self.train_dates = X_temp.index
        self.test_dates = X_test.index

        X_temp: np.array = self.scaler.fit_transform(X_temp)
        X_test: np.array = self.scaler.transform(X_test)

        self.test_generator = TimeseriesGenerator(
            X_test, y_test, length=window_size, batch_size=batch_size
        )

        if self.CV:  # cross-validation
            for train_index, val_index in self.tscv.split(X_temp):
                # Create generators for each split
                train_generator = TimeseriesGenerator(
                    X_temp[train_index], y_temp[train_index],
                    length=window_size,
                    batch_size=batch_size
                )
                val_generator = TimeseriesGenerator(
                    X_temp[val_index], y_temp[val_index],
                    length=window_size,
                    batch_size=batch_size
                )

                self.train_generators.append(train_generator)
                self.val_generators.append(val_generator)
        else:  # no cross-validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size, shuffle=False
            )
            self.train_generators.append(TimeseriesGenerator(
                X_train, y_train, length=window_size, batch_size=batch_size
            ))
            self.val_generators.append(TimeseriesGenerator(
                X_val, y_val, length=window_size, batch_size=batch_size
            ))


def train_RNN(
        future: str,
        data_params: dict[str, any],
        model_params: dict[str, any],
        rnn_type: str,
        max_epochs: int
    ) -> tuple[Sequential, RNNGenerator, dict[str, any]]:

    # Configure early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    # Create generatorW
    gen = RNNGenerator(
        future=future,
        CV=data_params['CV']
    )

    gen.preprocess_data(
        data_params['feature_columns'],
        data_params['target_column'],
        data_params['window_size'],
        test_size=data_params['test_size'],
        val_size=data_params['val_size'],
        scaler_type=data_params['scaler_type']
    )

    # Build the model
    model = build_rnn_model(
        rnn_type,
        model_params,
        (data_params['window_size'], len(data_params['feature_columns']))
    )

    _, ax = plt.subplots(figsize=(7, 5), dpi=200)

    history_list = list()

    # Train the model with early stopping and Cross Validation
    if len(gen.train_generators) > 1:
        loop_range = tqdm(
            range(len(gen.train_generators)),
            desc='Running Cross Validation',
        )
    else:
        loop_range = [0]

    for i in loop_range:
        history = model.fit(
            gen.train_generators[i],
            epochs=max_epochs,
            batch_size=model_params['batch_size'],
            validation_data=gen.val_generators[i],
            callbacks=[early_stopping],
            verbose=1
        )

        loss_dict = {
            'train_loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        }

        history_list.append(history)
        ax.plot(loss_dict['train_loss'], label=f'Train Loss k={i + 1}')
        ax.plot(loss_dict['val_loss'], linestyle=':', label=f'Val Loss k={i + 1}')

    ax.set_yscale('log')
    ax.set_title('Model Training and Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss MSE')
    ax.legend(frameon=False)

    return model, gen, loss_dict


def optimize_hyperparameters(
        future: str,
        trial_config: dict[str, any],
        data_params: dict[str, any],
        rnn_type: str,
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

    def objective(
            trial,
            config: dict[str, any]
        ):
        # Model configuration based on trial suggestions
        units_first_layer = trial.suggest_categorical(
            'units_first_layer', config['units_first_layer']
        )
        units_second_layer = trial.suggest_categorical(
            'units_second_layer', config['units_second_layer']
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

        # Building the model
        model = build_rnn_model(
            rnn_type,
            {
                'units_first_layer': units_first_layer,
                'units_second_layer': units_second_layer,
                'l2_strength': l2_strength,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'noise_std': noise_std
            },
            (data_params['window_size'], len(data_params['feature_columns']))
        )

        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True
        )

        gen = RNNGenerator(future=future, CV=data_params['CV'])

        gen.preprocess_data(
            data_params['feature_columns'],
            data_params['target_column'],
            data_params['window_size'],
            test_size=data_params['test_size'],
            val_size=data_params['val_size']
        )

        # Train the model
        history = model.fit(
            gen.train_generators[-1],
            epochs=50,
            batch_size=int(batch_size),
            validation_data=gen.val_generators[-1],
            callbacks=[early_stopping],
            verbose=0
        )

        return np.min(history.history['val_loss'])

    study = optuna.create_study(
        direction="minimize",
        study_name=f'{rnn_type}_study'
    )

    study.optimize(
        partial(objective, config=trial_config),
        n_trials=n_trials,
        n_jobs=n_jobs
    )

    return study.best_params


def save_model_info(
        model: Sequential,
        model_name: str,
        model_params: dict[str, any],
        data_params: dict[str, any],
        loss_dict: dict[str, any]
    ) -> None:

    if not os.path.exists(f'model_archive/{model_name}'):
        os.makedirs(f'model_archive/{model_name}')

        model.save(f'model_archive/{model_name}/model_weights.h5')

        with open(
            f'model_archive/{model_name}/model_params.json', 'w'
            ) as file:
            json.dump(model_params, file, indent=4)

        with open(
            f'model_archive/{model_name}/data_params.json', 'w'
            ) as file:
            json.dump(data_params, file, indent=4)

        with open(
            f'model_archive/{model_name}/loss_data.json', 'w'
            ) as file:
            json.dump(loss_dict, file, indent=4)

        print(f'Model saved as {model_name}')
    else:
        print('Model already exists')
