import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification


def get_BERT_sentiment(
        sentence_list: list[str], 
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