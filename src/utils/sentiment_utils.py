import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from tqdm.auto import tqdm

from typing import Union


def add_textblob_polarity(
        text_series: pd.Series,
        name: Union[None, str] = None
    ) -> pd.Series:
    """
    Add TextBlob polarity scores to a text series

    Parameters
    ----------
        text_series : pd.Series
            Series of text to be analyzed

    Returns
    -------
        pd.Series
            Series of polarity scores
    """
    object_name = text_series.name if name is None else name
    tqdm.pandas(desc=f"Textblob progress for {object_name}")
    polarity_scores = text_series.progress_apply(
        lambda x: TextBlob(x).sentiment.polarity
    )

    return polarity_scores


def add_vader_compound(
        text_series: pd.Series,
        name: Union[None, str] = None
    ) -> pd.Series:
    """
    Add VADER compound scores to a text series

    Parameters
    ----------
        text_series : pd.Series
            Series of text to be analyzed

    Returns
    -------
        pd.Series
            Series of compound scores
    """
    object_name = text_series.name if name is None else name
    tqdm.pandas(desc=f"VADER progress for {object_name}")
    vader = SentimentIntensityAnalyzer()
    compound_scores = text_series.progress_apply(
        lambda x: vader.polarity_scores(x)['compound']
    )

    return compound_scores
