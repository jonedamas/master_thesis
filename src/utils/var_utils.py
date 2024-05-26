import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


def plot_criterion(
        lag_orders: int,
        ax: plt.Axes,
        name: str
    ) -> None:

    colors = ['crimson', 'navy', 'limegreen']

    for k, ic in enumerate(['aic', 'bic', 'hqic']):
        ic_info = lag_orders.ics[ic]
        lags = range(len(ic_info))
        ax.plot(lags, ic_info, label=ic.upper(), color=colors[k], lw=0.8)

        min_ic = np.argmin(ic_info)
        ax.plot(min_ic, ic_info[min_ic], 'ro', color=colors[k])
        # annotate the min point
        ax.annotate(
            f'{min_ic}', (min_ic, ic_info[min_ic]),
            textcoords="offset points", xytext=(0, 10),
            ha='center', fontsize=10
        )

    ax.set_xlabel('Lag order')
    ax.legend(frameon=False, loc='upper center', title=name, fontsize=10)


def grangers_causation_matrix(
        data: pd.DataFrame,
        variables: list[str],
        test: str='ssr_chi2test',
        verbose: bool=False,
        maxlag: int=12
    ) -> pd.DataFrame:

    df = pd.DataFrame(
        np.zeros((len(variables), len(variables))),
        columns=variables,
        index=variables
    )
    for c in tqdm(df.columns, desc='Granger Causality Matrix'):
        for r in df.index:
            test_result = grangercausalitytests(
                data[[r, c]],
                maxlag=maxlag,
                verbose=False
            )
            p_values = [round(test_result[i+1][0][test][1], 4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]

    return df


class SentVAR:
    def __init__(
            self,
            dfs: dict[str, pd.DataFrame],
            topic: str,
            analyzer: str,
            lags: int = 19
        ):
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

    def plot_irf(
            self,
            ax: plt.Axes
        ) -> None:

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
