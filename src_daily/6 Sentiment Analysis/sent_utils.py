from julia import Main

import pandas as pd
from typing import List

import torch.nn.functional as F
import torch
from nltk.tokenize import sent_tokenize
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

    KwArgs:
        frequency: str

    Returns:
        pd.DataFrame
    """
    resampled_df = df[colnames].resample(frequency).mean().dropna()

    date_range = pd.date_range(start=resampled_df.index.min(), end=resampled_df.index.max(), freq='D')

    resampled_df = resampled_df.reindex(date_range, fill_value=0)

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
    sent_series = Main.SI_bai(sent_series.to_list(), float(beta))

    return sent_series