import pandas as pd
import requests

from bs4 import BeautifulSoup
from newspaper import Article
from tqdm import tqdm
from datetime import datetime



# Get URL df from google news
def get_url_gnews(url: str) -> pd.DataFrame:

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    time_elements = [datetime.fromisoformat(i['datetime'].replace('Z', '+00:00')) for i in soup.find_all('time', class_='hvbAAd')]

    link_list = [f'https://news.google.com{i["href"][1:]}' for i in soup.find_all('a', class_='WwrzSb')]

    df = pd.DataFrame({'link': link_list, 'datetime': time_elements})

    df.index = df['datetime']
    df = df.drop('datetime', axis=1)

    return df


# Create a dataframe with the news
def create_news_df(df: pd.DataFrame) -> pd.DataFrame:

    info_dict = {
        'title': [], # Title of the article
        'text': [], # Text of the article
        'url': [], # URL of the article
        'date': [] # Date of the article
    }

    for row in tqdm(df.iterrows()):
        url = row[1]['link']
        try:
            article = Article(url)
            article.download()
            article.parse()

            info_dict['title'].append(article.title)
            info_dict['text'].append(article.text)
            info_dict['url'].append(url)
            info_dict['date'].append(row[0])
        except:
            continue

    df = pd.DataFrame(info_dict, index=info_dict['date'])
    df.drop('date', axis=1, inplace=True)

    return df



