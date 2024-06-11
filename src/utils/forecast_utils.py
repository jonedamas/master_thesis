import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm.notebook import tqdm

from typing import List, Tuple, Dict
import json
import os
import sys
from dotenv import load_dotenv
load_dotenv()
REPO_PATH= os.getenv('REPO_PATH')
sys.path.insert(0, rf'{REPO_PATH}src')

from utils.model_utils import build_rnn_model, RNNGenerator
from utils.eval_utils import calculate_metrics
from utils.var_utils import forecast_var


class ForecastPredictions:
    def __init__(self, model_name: str):
        """
        Load the predictions from a forecast model

        Parameters
        ----------
        model_name
            The name of the model to load
        """
        self.model_name = model_name

        if model_name.split('_')[1] == 'VAR':
            self.results = forecast_var(model_name)
        else:
            self.results = forecast_rnn(model_name)

        self.model = self.results['model']
        self.y_test = self.results['y_test']
        self.y_pred = self.results['y_pred']

    def metrics(
            self,
            decimals: float = 4,
            print_metrics: bool = False
        ) -> None | dict[str, any]:
        """
        Calculate the mean squared error, mean absolute error,
        and mean directional accuracy.

        Parameters
        ----------
        decimals
            The number of decimal places to round the metrics to
        print_metrics
            Whether to print the metrics or return them

        Returns
        -------
        None | dict[str, any]
            Metrics for the forecast model
        """
        metrics = calculate_metrics(
            self.y_test,
            self.y_pred,
            decimals=decimals
        )

        if print_metrics:
            print(f'Model: {self.model_name}')
            for name, metric in metrics.items():
                print(f'{name.upper()}: {metric}')
        else:
            return metrics

    def dm_test(
            self,
            bm_model: 'ForecastPredictions',
            crit: str = 'MSE'
        ) -> Dict[str, float]:
        """
        Perform the Diebold-Mariano test on the model and the benchmark model.

        Parameters
        ----------
        bm_model
            The benchmark model to compare to
        crit
            The criterion to use for the test

        Returns
        -------
        dict[str, float]
            The Diebold-Mariano test statistic and p-value
        """

        if len(self.y_test) != len(bm_model.y_test):
            diff = abs(len(self.y_test) - len(bm_model.y_test))
            if len(self.y_test) > len(bm_model.y_test):
                self.y_test = self.y_test[diff:]
                self.y_pred = self.y_pred[diff:]
            else:
                bm_model.y_test = bm_model.y_test[diff:]
                bm_model.y_pred = bm_model.y_pred[diff:]

        e1 = np.array(bm_model.y_test - bm_model.y_pred)
        e2 = np.array(self.y_test - self.y_pred)

        if crit == "MSE":
            d = e1**2 - e2**2
        elif crit == "MAE":
            d = np.abs(e1) - np.abs(e2)
        else:
            raise ValueError("Criterion must be 'MSE' or 'MAE'")

        mean_d = np.mean(d)
        var_d = np.var(d, ddof=1)

        DM_stat = mean_d / np.sqrt(var_d / len(d))
        p_value = 2 * (1 - norm.cdf(np.abs(DM_stat)))

        dm_dict = {
            'DM_stat': round(DM_stat, 4),
            'p_value': round(p_value, 4)
        }

        return dm_dict


def load_models(
    model_names: List[str],
    benchmark_forecast: ForecastPredictions,
    dm_crit: str = 'MSE'
    ) -> Tuple[Dict, pd.DataFrame]:
    """
    Load the forecast models and calculate the metrics for each model.

    Parameters
    ----------
    model_names
        The names of the models to load
    benchmark_forecast
        The benchmark forecast to compare to
    dm_crit
        The criterion to use for the Diebold-Mariano test

    Returns
    -------
    Tuple[Dict, pd.DataFrame]
        A dictionary of the models and a DataFrame of the metrics
    """

    metric_list = []
    model_dict = dict()
    for model_name in tqdm(model_names, desc='Loading models'):

        fc_model = ForecastPredictions(model_name)
        model_dict[model_name] = fc_model
        metric_dict = fc_model.metrics(decimals=4)
        metric_dict.update(fc_model.dm_test(benchmark_forecast, dm_crit))
        metric_list.append(metric_dict)

    metric_df = pd.DataFrame(metric_list, index=model_names)

    return model_dict, metric_df


def forecast_rnn(model_name: str) -> Dict[str, any]:
    """
    Forecast using an RNN model.

    Parameters
    ----------
    model_name
        The name of the RNN model to forecast with

    Returns
    -------
    Dict[str, any]
        The forecast results
    """
    with open(
        f'model_archive/{model_name}/data_params.json',
        'r') as f:
        data_params = json.load(f)
    with open(
        f'model_archive/{model_name}/model_params.json',
        'r') as f:
        model_params = json.load(f)

    model_comp = model_name.split('_')
    future = model_comp[0]
    rnn_type = model_comp[1]

    gen = RNNGenerator(future)

    gen.preprocess_data(
        data_params['feature_columns'],
        data_params['target_column'],
        model_params['window_size'],
        test_size=data_params['test_size'],
        val_size=data_params['val_size'],
    )

    model = build_rnn_model(rnn_type, model_params,
        (
            model_params['window_size'],
            len(data_params['feature_columns'])
        )
    )

    model.load_weights(
        f'model_archive/{model_name}/model_weights.h5'
    )

    test_predictions = model.predict(
        gen.test_generator
    ).flatten()

    test_targets = np.concatenate(
        [y for _, y in gen.test_generator]
    ).flatten()

    results = {
        'y_test': test_targets,
        'y_pred': test_predictions,
        'model': model
    }

    return results
