import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk_stop_words = set(stopwords.words('english'))

ignore_words = set(
    [
        'Full', 'Story', 'Reuters', 'copyright', 'c', 'Thomson', 'Click', 'Restrictions',
        'Thomson Reuters', 'Full Story', 'Click Restrictions', 'c Copyright', 'Copyright Thomson',
        'Restrictions https', 'Reuters Click', 'Final Terms', 'https'
    ]
)

stop_words = nltk_stop_words.union(ignore_words)


def clean_tokens(
        tokens: list[str]
    ) -> list[str]:
    """
    Clean a list of tokens by removing punctuation and stopwords.

    Parameters
    ----------
        tokens: list[str]
            The list of tokens to clean.

    Returns
    -------
        list[str]: The cleaned list of tokens.
    """
    tokens_wo_punct: list = [word for word in tokens if word.isalnum()]
    tokens_wo_sw: list = [word for word in tokens_wo_punct if word.lower() not in stop_words]

    return tokens_wo_sw


def clean_token_series(
        series: pd.Series
    ) -> pd.DataFrame:
    """
    Tokenize and clean a pandas Series of text.

    Parameters
    ----------
        series: pd.Series
            The pandas Series of text to tokenize and clean.

    Returns
    -------
        pd.DataFrame: A DataFrame with two columns: 'tokenized' and 'cleaned'.

    """
    tokenized: pd.Series = series.apply(word_tokenize)
    cleaned_tokenized: pd.Series = tokenized.apply(clean_tokens)

    return tokenized, cleaned_tokenized
