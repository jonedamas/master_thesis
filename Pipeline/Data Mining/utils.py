import pandas as pd
import requests

from bs4 import BeautifulSoup
from newspaper import Article
from tqdm import tqdm


# Get URL list from google news
def get_url_gnews(url: str) -> list:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all links
    article_list = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('./articles/'):
            full_url = f'https://news.google.com{href[1:]}'
            article_list.append(full_url)

    # Remove duplicates
    article_list = list(set(article_list))

    return article_list


def create_news_df(url_list: list) -> pd.DataFrame:

    info_dict = {
        'title': [], # Title of the article
        'text': [], # Text of the article
        'url': [], # URL of the article
        'date': [] # Date of the article
    }

    for URL in tqdm(url_list):
        try:
            article = Article(URL)
            article.download()
            article.parse()

            info_dict['title'].append(article.title)
            info_dict['text'].append(article.text)
            info_dict['url'].append(URL)
            info_dict['date'].append(article.publish_date)
        except:
            continue

    print(f'{len(info_dict["url"])} of {len(url_list)} articles scraped')

    return pd.DataFrame(info_dict)