import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from tqdm.auto import tqdm


def add_textblob_polarity(
        text_series: pd.Series
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
    tqdm.pandas(desc=f"Textblob progress for {text_series.name}")
    polarity_scores: pd.Series = text_series.progress_apply(
        lambda x: TextBlob(x).sentiment.polarity
    )

    return polarity_scores


def add_vader_compound(
        text_series: pd.Series
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
    tqdm.pandas(desc=f"VADER progress for {text_series.name}")
    vader: SentimentIntensityAnalyzer = SentimentIntensityAnalyzer()
    compound_scores: pd.Series = text_series.progress_apply(
        lambda x: vader.polarity_scores(x)['compound']
    )

    return compound_scores
