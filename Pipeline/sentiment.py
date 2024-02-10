import numpy as np

from transformers import BertTokenizer, BertForSequenceClassification

finbert = BertForSequenceClassification.from_pretrained(
    'yiyanghkust/finbert-tone',
    num_labels=3
)

tokenizer = BertTokenizer.from_pretrained(
    'yiyanghkust/finbert-tone'
)

labels: dict = {
    0:'neutral', 
    1:'positive', 
    2:'negative'
}

def get_sentiment(sentence_list: list[str]):
    
    inputs = tokenizer(
        sentence_list, 
        return_tensors="pt", 
        padding=True
    )

    outputs = finbert(**inputs)[0]

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