import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm.notebook import tqdm

from typing import Callable, List, Tuple, Dict, Union
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
    def __init__(self, model_name: str, forecast_func: Callable):
        self.model_name = model_name
        self.results = forecast_func(model_name)

        self.model = self.results['model']
        self.y_test = self.results['y_test']
        self.y_pred = self.results['y_pred']

    def metrics(
            self,
            decimals: float = 4,
            print_metrics: bool = False
        ) -> None | dict[str, any]:

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

        # remove the difference allowing comparison between RNN and other models
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
    benchmark_forecast: ForecastPredictions
    ) -> Tuple[Dict, pd.DataFrame]:

    metric_list = []
    model_dict = dict()
    for model_name in tqdm(model_names, desc='Loading models'):
        model_type = model_name.split('_')[1]

        if model_type == 'VAR':
            func = forecast_var
        else:
            func = forecast_rnn

        fc_model = ForecastPredictions(model_name, func)
        model_dict[model_name] = fc_model
        metric_dict = fc_model.metrics(decimals=4)
        metric_dict.update(fc_model.dm_test(benchmark_forecast))
        metric_list.append(metric_dict)

    metric_df = pd.DataFrame(metric_list, index=model_names)

    return model_dict, metric_df


def forecast_rnn(
    model_name: str
    ) -> Dict[str, any]:

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

    gen = RNNGenerator(
        future
    )

    gen.preprocess_data(
        data_params['feature_columns'],
        data_params['target_column'],
        data_params['window_size'],
        test_size=data_params['test_size'],
        val_size=data_params['val_size'],
    )

    # Build the model
    model = build_rnn_model(
        rnn_type,
        model_params,
        (
            data_params['window_size'],
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
