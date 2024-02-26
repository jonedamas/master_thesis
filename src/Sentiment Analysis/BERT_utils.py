import numpy as np
import pandas as pd
from typing import List
import torch
from transformers import BertTokenizer, BertForSequenceClassification


def get_BERT_sentiment(
        sentence_list: List[str], 
        model: BertForSequenceClassification, 
        tokenizer: BertTokenizer
        ) -> dict[str, float]:
    '''
    Function to get the sentiment of a list of sentences using a BERT model.

    Args:
        sentence_list: list of strings, the sentences to be analyzed.
        model: BertForSequenceClassification, the BERT model to be used.
        tokenizer: BertTokenizer, the tokenizer to be used.

    Returns:
        dict[str, float]: a dictionary with the sentiment of each sentence.
    '''

    inputs = tokenizer(
        sentence_list, 
        return_tensors="pt", 
        padding=True
    )

    outputs = model(**inputs)[0]

    res = {}

    for i, sentence in enumerate(sentence_list):
        res[i] = [np.argmax(outputs.detach().numpy()[i]), sentence]

    df = pd.DataFrame.from_dict(
        res, 
        orient='index', 
        columns=['prediction', 'sentence']
    )

    compounded_score = df.prediction.value_counts(normalize=True).to_dict()

    return compounded_score


def get_BERT_sentiment_per_headline(
    sentence_series: pd.Series, 
    model: BertForSequenceClassification, 
    tokenizer: BertTokenizer
) -> List[float]:
    '''
    Function to get the sentiment score of a list of headlines using a BERT model.

    Args:
        sentence_list: List of strings, the headlines to be analyzed.
        model: BertForSequenceClassification, the BERT model to be used.
        tokenizer: BertTokenizer, the tokenizer to be used.

    Returns:
        List[float]: A list of sentiment scores for each headline.
    '''

    inputs = tokenizer(
        sentence_series.to_list(), 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=512
        )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs).logits

    probabilities = torch.softmax(outputs, dim=1)

    positive_scores = probabilities[:, 1].cpu().numpy()

    negative_scores = probabilities[:, 0].cpu().numpy()

    compound_scores = positive_scores - negative_scores

    return compound_scores.tolist()