from bs4 import BeautifulSoup
import eikon as ek
import pandas as pd
import json

import os
from dotenv import load_dotenv

load_dotenv()
ek.set_app_key(os.getenv('EIKON_API_KEY'))
repo_path = os.getenv('REPO_PATH')


def get_story_text(
        id: str,
        text_dict: dict[str:str]
    ) -> None:
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
    soup: BeautifulSoup = BeautifulSoup(response, 'html.parser')
    text: str = soup.get_text()
    text_dict[id] = text


def load_previous_stories(
        headline_topic: str
    ) -> dict[str, str]:
    """
    Load the previous stories from a file.

    Parameters
    ----------
        headline_topic: the topic of the headlines to load

    Returns
    -------
        The previous stories as a dictionary
    """
    file_path: str = repo_path + rf'data\raw_news_stories\EIKON_{headline_topic}_NEWS_FULL.json'

    # check if file exists
    if not os.path.exists(file_path):
        json.dump({}, open(file_path, 'w'))

    with open(file_path, 'r') as f:
        previous_stories: dict[str:str] = json.load(f)

    return previous_stories
