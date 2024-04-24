import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from tqdm.auto import tqdm
import torch.nn.functional as F

FINBERT_MODEL = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels = 3)
FINBERT_TOKENIZER = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')


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


def finbert_sentiment(
        text: str,
    ) -> float:
    """
    Calculate the sentiment score of a text using FinBERT

    Parameters
    ----------
        text : str
            Text to be analyzed

    Returns
    -------
        float
            Sentiment score
    """
    inputs = FINBERT_TOKENIZER(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True,  # The tokenizer will truncate the text so that it fits the model's maximum input size.
        padding=True,  # The tokenizer will add padding to the text if it is shorter than the model's maximum input length
        )

    with torch.no_grad():
        prediction = FINBERT_MODEL(**inputs)

    probabilities: torch.Tensor = F.softmax(prediction.logits, dim=-1).squeeze()

    negative, _, positive = probabilities
    score: float = (positive - negative).item() # calculate compound score

    return score


def add_finbert_compound(
        text_series: pd.Series
    ) -> pd.Series:
    """
    Add FinBERT compound scores to a text series

    Parameters
    ----------
        text_series : pd.Series
            Series of text to be analyzed

    Returns
    -------
        pd.Series
            Series of compound scores
    """
    results: list = []

    tqdm.pandas(desc=f"FinBERT progress for {text_series.name}")
    for text in tqdm(text_series):
        finbert_score: float = finbert_sentiment(text)

        results.append(finbert_score)

    result_series: pd.Series = pd.Series(results)

    return result_series
