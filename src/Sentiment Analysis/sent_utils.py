import pandas as pd
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch
from typing import List
from textblob import TextBlob
from transformers import BertTokenizer, BertForSequenceClassification


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

    Returns:
        pd.DataFrame
    """
    resampled_df = df[colnames].resample(frequency).mean().dropna()

    return  resampled_df


def FinBERT_sentiment(
        headline: pd.Series,
        model: BertForSequenceClassification,
        tokenizer: BertTokenizer
    ) -> pd.Series:

    input = tokenizer(headline, return_tensors="pt")

    with torch.no_grad():
        prediction = model(**input)

    probabilities = F.softmax(prediction.logits, dim=-1)

    negative, neutral, positive = probabilities[0]
    score = (positive - negative).item()

    return score


def textblob_sentiment_df(
        df: pd.DataFrame,
        index_function,
        beta: float,
        frequency: str = 'D'
    ) -> pd.DataFrame:
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

    df['polarity'] = df['polarity'] ** 2

    sent_df = aggregate_score(df, ['polarity', 'subjectivity'], frequency=frequency)

    sent_df['SV'] = index_function(list(sent_df['polarity'].values), beta)

    return sent_df
