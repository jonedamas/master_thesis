from julia import Main
import pandas as pd

Main.include(r'sent_index.jl')


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
