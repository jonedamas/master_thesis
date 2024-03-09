from julia import Main

import pandas as pd
import torch.nn.functional as F
import torch
from typing import List
from textblob import TextBlob
from transformers import BertTokenizer, BertForSequenceClassification

Main.include(r'sent_index.jl')

FinBERT_model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels = 3)
FinBERT_tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

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
    ) -> pd.Series:

    input = FinBERT_tokenizer(headline, return_tensors="pt")

    with torch.no_grad():
        prediction = FinBERT_model(**input)

    probabilities = F.softmax(prediction.logits, dim=-1)

    negative, neutral, positive = probabilities[0]
    score = (positive - negative).item()

    return score


def textblob_sentiment(
        headline: pd.Series,
    ) -> pd.Series:
    """
    Returns a series with the sentiment scores of the given series

    Args:
        df: pd.Series

    Returns:
        pd.Series
    """
    textblob_series = headline.apply(
        lambda x: pd.Series(TextBlob(x).sentiment[0])
    )

    return textblob_series


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
    sent_series = Main.SI_bai(sent_series.to_list(), beta)

    return sent_series


def SI_jone(
        sent_series: pd.Series,
        beta: float,
        lambda_: float
    ) -> pd.Series:
    """
    Returns the sentiment index of the given sentiment series

    Args:
        sent_series: pd.Series
        beta: float

    Returns:
        pd.Series
    """
    sent_series = Main.SI_jone(sent_series.to_list(), beta, lambda_)

    return sent_series
