import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

NLTK_STOP_WORDS = set(stopwords.words('english'))

IGNORE_WORDS = set(
    [
        'Full', 'Story', 'Reuters', 'copyright', 'c', 'Thomson', 'Click', 'Restrictions',
        'Thomson Reuters', 'Full Story', 'Click Restrictions', 'c Copyright', 'Copyright Thomson',
        'Restrictions https', 'Reuters Click', 'Final Terms', 'https'
    ]
)

stop_words = NLTK_STOP_WORDS.union(IGNORE_WORDS)


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
        series: pd.Series,
        include_raw: bool = False
    ) -> pd.Series | tuple[pd.Series, pd.Series]:
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
    tqdm.pandas(desc=f"Tokenization progress for {series.name}")
    tokenized: pd.Series = series.progress_apply(word_tokenize)
    tqdm.pandas(desc=f"Cleaning progress for {series.name}")
    cleaned_tokenized: pd.Series = tokenized.progress_apply(clean_tokens)

    if include_raw:
        return tokenized, cleaned_tokenized
    else:
        return cleaned_tokenized



def create_wordcloud(
        word_series: pd.Series,
        width: int = 400,
        height: int = 400,
    ) -> WordCloud:
    """
    Create a WordCloud object from a pandas Series of words.

    Parameters
    ----------
        word_series_: pd.Series
            The pandas Series of words to create the WordCloud from.
        width: int
            The width of the WordCloud.
        height: int
            The height of the WordCloud.

    Returns
    -------
        WordCloud: The WordCloud object created from the word_series.
    """
    wordcloud: WordCloud = WordCloud(
        width=width,
        height=height,
        max_font_size=100,
        max_words=100,
        colormap='twilight',
        background_color='white'
    ).generate(' '.join(word_series.to_list()))

    return wordcloud
