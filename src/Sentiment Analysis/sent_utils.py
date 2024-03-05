import pandas as pd
from typing import List
from textblob import TextBlob

def aggregate_score(df: pd.DataFrame, colnames: List[str], frequency: str = 'D') -> pd.DataFrame:
    """
    Aggregates the sentiment scores of the given columns in the dataframe

    Args:
        df: pd.DataFrame
        colnames: list[str]

    Returns:
        pd.DataFrame
    """
    resampled_df = df[colnames].resample(frequency).mean().dropna()

    return  resampled_df


def textblob_sentiment_df(df: pd.DataFrame, index_function, frequency: str = 'D') -> pd.DataFrame:
    """
    Returns a dataframe with the sentiment scores of the given dataframe

    Args:
        df: pd.DataFrame
        index_function: function
        frequency: str

    Returns:
        pd.DataFrame
    """
    df[['polarity', 'subjectivity']] = df['headline'].apply(lambda x: pd.Series(TextBlob(x).sentiment))

    sent_df = aggregate_score(df, ['polarity', 'subjectivity'], frequency=frequency)

    sent_df['SV'] = index_function(list(sent_df['polarity'].values))

    return sent_df
