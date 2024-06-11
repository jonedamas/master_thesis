import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from typing import List, Tuple, Dict, Union
import os
import sys
import json
import warnings
from dotenv import load_dotenv

load_dotenv()
REPO_PATH= os.getenv('REPO_PATH')
sys.path.insert(0, rf'{REPO_PATH}src')

from utils.main_utils import load_processed

warnings.filterwarnings("ignore")


def plot_criterion(lag_orders: int, ax: plt.Axes, name: str) -> None:
    """
    Plot the AIC, BIC, and HQIC criterion for the given lag orders.

    Parameters
    ----------
    lag_orders
        The lag orders to plot
    ax
        The axis to plot on
    name
        The name of the model
    """

    colors = ['crimson', 'navy', 'limegreen']

    for k, ic in enumerate(['aic', 'bic', 'hqic']):
        ic_info = lag_orders.ics[ic][1:]
        lags = range(len(ic_info))
        ax.plot(lags, ic_info, label=ic.upper(), color=colors[k], lw=0.8)

        min_ic = np.argmin(ic_info)
        ax.plot(min_ic, ic_info[min_ic], 'ro', color=colors[k])
        ax.annotate(
            f'{min_ic}', (min_ic, ic_info[min_ic]),
            textcoords="offset points", xytext=(0, 10),
            ha='center', fontsize=10
        )

    ax.set_xlabel('Lag order')
    ax.legend(frameon=False, loc='upper center', title=name, fontsize=10)


def grangers_causation(
        data: pd.DataFrame,
        variables: List[str],
        target: str,
        test: str = 'ssr_chi2test',
        verbose: bool = False,
        maxlag: int = 4
    ) -> pd.DataFrame:
    """
    Perform the Granger Causality Test on the given data.

    Parameters
    ----------
    data
        The DataFrame containing the data
    variables
        The list of variables to test
    target
        The target variable
    test
        The test to perform
    verbose
        Whether to print the results
    maxlag
        The maximum lag to test

    Returns
    -------
    pd.DataFrame
        The DataFrame containing the p-values of the test
    """

    p_values_df = pd.DataFrame(
        np.zeros((maxlag, len(variables))),
        columns=variables,
        index=[f'lag_{i+1}' for i in range(maxlag)]
    )

    for r in tqdm(variables, desc='Granger Causality Test'):
        test_result = grangercausalitytests(
            data[[r, target]],
            maxlag=maxlag,
            verbose=verbose
        )
        p_values = [round(test_result[i+1][0][test][1], 4) for i in range(maxlag)]
        if verbose:
            print(f'Y = {target}, X = {r}, P Values = {p_values}')
        p_values_df[r] = p_values

    return p_values_df.T


class SentVAR:
    def __init__(
            self,
            dfs: Dict[str, pd.DataFrame],
            topic: str,
            analyzer: str,
            lags: int = 19
        ):
        """
        Set up the VAR model for structural sentiment analysis.

        Parameters
        ----------
        dfs
            The DataFrames to use
        topic
            The topic of the model
        analyzer
            The sentiment analyzer used
        lags
            The number of lags to use
        """
        self.dfs = dfs.copy()
        self.topic = topic

        self.sent_col = f'{topic}_{analyzer}'

        self.model_dict = {}

        for key, df in self.dfs.items():
            df['positive'] = df[self.sent_col].apply(
                lambda x: x if x > 0 else 0
            )
            df['negative'] = df[self.sent_col].apply(
                lambda x: -x if x < 0 else 0
            )

            # VAR model
            model: VAR = VAR(
                df[['positive', 'negative', 'REALIZED_VOL']]
            )
            results = model.fit(lags)
            irf = results.irf(100)

            fig = irf.plot(
                orth=False,
                response='REALIZED_VOL'
            )
            plt.close()

            self.model_dict[key] = {
                'model': model,
                'results': results,
                'irf': irf,
                'fig': fig,
                'axes_list': fig.axes[:2]
            }

    def plot_irf(self, ax: plt.Axes) -> None:

        linestyles = {'CLc1': '-', 'LCOc1': '--'}

        colors = ['#056517', '#bf1029']
        labels = ['Positive', 'Negative']

        for key, model_info in self.model_dict.items():
            axes_list = model_info['axes_list']

            for i, plot_ax in enumerate(axes_list):
                ax.plot(
                    plot_ax.lines[0].get_xdata(),
                    plot_ax.lines[0].get_ydata(),
                    color=colors[i],
                    label=f'{labels[i]} - {key}',
                    linestyle=linestyles[key],
                )

        ax.set_title(rf'{self.topic} $\rightarrow$ Realized Volatility', fontsize=17)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_xlabel('Hours after shock', fontsize=16)
        ax.set_ylabel('Response', fontsize=16)
        x_ticks = np.arange(0, 101, 20)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels((x_ticks * 5 / 60).astype(int))


def forecast_var(
    model_name: str,
    var_params: Union[None, Dict[str, any]] = None,
    return_model: bool = False
    ) -> Union[dict[str, any], Tuple[dict[str, any], VAR]]:
    """
    Forecast the VAR model.

    Parameters
    ----------
    model_name
        The name of the model
    var_params
        The parameters of the model
    return_model
        Whether to return the model

    Returns
    -------
    dict[str, any] | Tuple[dict[str, any], VAR]
        The results of the forecast
    """
    if var_params is None:
        with open(
            f'model_archive/{model_name}/var_params.json',
            'r') as f:
            var_params = json.load(f)

    future = model_name.split('_')[0]
    lags = var_params['lags']
    features = var_params['features']
    target = var_params['target']

    df = load_processed(future)[future]

    train, test = train_test_split(
        df[features], test_size=var_params['test_size'], shuffle=False
    )

    model = VAR(train)

    results = model.fit(lags)

    forecasted_values = []
    for i in tqdm(range(len(test)), desc='Forecasting with VAR'):
        if i == 0:
            input_data = train.values[-lags:]
        else:
            input_data = df[features].iloc[len(train) + i - lags:len(train) + i].values

        forecast = results.forecast(input_data, steps=1)
        forecasted_values.append(forecast[0][0])

    y_test = np.array(test[target])
    y_pred = np.array(forecasted_values)

    results = {
        'y_test': y_test,
        'y_pred': y_pred,
        'model': model
    }

    if return_model:
        return results, model
    else:
        return results


def save_var_info(model_name: str, var_params: Dict[str, any]) -> None:
    """
    Save the VAR model information.

    Parameters
    ----------
    model_name
        The name of the model
    var_params
        The parameters of the model
    """
    if not os.path.exists(f'model_archive/{model_name}'):
        os.makedirs(f'model_archive/{model_name}')

        with open(
            f'model_archive/{model_name}/var_params.json', 'w'
            ) as file:
            json.dump(var_params, file, indent=4)

        print(f'Model saved as {model_name}')
    else:
        print('Model already exists')
