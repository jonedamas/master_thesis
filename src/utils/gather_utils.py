from bs4 import BeautifulSoup
import eikon as ek
import pandas as pd
from tqdm import tqdm

import json
from typing import Dict
import os
from dotenv import load_dotenv

load_dotenv()
ek.set_app_key(os.getenv('EIKON_API_KEY'))
REPO_PATH= os.getenv('REPO_PATH')


def get_story_text(id: str, text_dict: Dict[str, str]) -> None:
    """
    Get the text of a news story from Eikon and store it in a dictionary.

    Parameters
    ----------
        id: the story ID to get the text for
        text_dict: the dictionary to store the text in

    Returns
    -------
        None
    """
    response = ek.get_news_story(id)
    soup = BeautifulSoup(response, 'html.parser')
    text = soup.get_text()
    text_dict[id] = text


def load_previous_stories(headline_topic: str) -> Dict[str, str]:
    """
    Load the previous stories from a file.

    Parameters
    ----------
        headline_topic: the topic of the headlines to load

    Returns
    -------
    Dict[str, str]
        The previous stories as a dictionary
    """
    file_path = rf'{REPO_PATH}data\raw_news_stories\EIKON_{headline_topic}_NEWS_FULL.json'

    if not os.path.exists(file_path):
        json.dump({}, open(file_path, 'w'))

    with open(file_path, 'r') as f:
        previous_stories = json.load(f)

    return previous_stories


error_types = {
    'limit_error': 'Error code 429 | Client Error: Too many requests, please try again later.',
    'backend_error': 'Error code 404 | Backend error. 404 Not Found'
}


def extract_stories(story_ids: pd.Series) -> Dict[str, str]:
    """
    Extract the text of news stories from Eikon.

    Parameters
    ----------
        storie_ids: the IDs of the stories to extract the text for

    Returns
    -------
    Dict[str, str]
        A dictionary containing the text of the stories
    """
    new_dict = {}

    for id in tqdm(story_ids):
        try:
            get_story_text(id, new_dict)

        except ek.EikonError as e:
            print(f'Error code:: {str(e)}')
            if str(e) == error_types['limit_error']:
                print('Daily request limit reached')
                break

            elif str(e) == error_types['backend_error']:
                new_dict[id] = 'error'

            else:
                new_dict[id] = 'error'

    print(f'Number of new stories downloaded: {len(new_dict)}')

    return new_dict
