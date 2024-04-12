import pandas as pd
import json
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from typing import Dict


def plot_news_frequency(
        df: pd.DataFrame,
        title: str,
        frequency: str = 'Q',
        **kwargs: Dict[str, str]
    ) -> plt.Figure:
    '''Plots the frequency of news articles over time. The frequency is aggregated by quarter.

    Args:
        df: pd.DataFrame
            A DataFrame containing the news articles and their publication dates.
        title: str
            The title of the plot.
    KwArgs:
        frequency: str
            The frequency of the aggregation. Default is 'Q' for quarter.

    Returns:
        fig: plt.Figure
            A matplotlib figure object containing the plot.
    '''

    fig, ax = plt.subplots(figsize=(10, 4), dpi=200)

    plt.rcParams['font.family'] = 'Arial'

    df['count'] = 1
    frequency = df['count'].resample(frequency).sum()

    ax.bar(frequency.index, frequency.values, width=75, **kwargs)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Date', fontsize=13)
    ax.set_ylabel('Number of articles', fontsize=13)

    # Only view years on x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)

    return fig
