from julia import Main

import pandas as pd
from typing import List

Main.include(r'sent_index.jl')


def aggregate_score(
        df: pd.DataFrame,
        colnames: List[str],
        frequency: str = 'D'
    ) -> pd.DataFrame:
    """
    Aggregates the sentiment scores of the given columns in the dataframe

    Args:
        df: pd.DataFrame
        colnames: list[str]

    KwArgs:
        frequency: str

    Returns:
        pd.DataFrame
    """
    df['count'] = 1

    resampled_df = df[colnames].resample(frequency).mean().dropna()
    resampled_df['count'] = df['count'].resample(frequency).sum()

    date_range = pd.date_range(start=resampled_df.index.min(), end=resampled_df.index.max(), freq=frequency)

    resampled_df = resampled_df.reindex(date_range, fill_value=0)

    return  resampled_df


def SI_bai(
        sent_series: pd.Series,
        beta: float
    ) -> pd.Series:
    """
    Returns the sentiment index of the given sentiment series

    Args:
        sent_series: pd.Series
        beta: float

    Returns:
        pd.Series
    """
    sent_series = Main.SI_bai(sent_series.to_list(), float(beta))

    return sent_series
