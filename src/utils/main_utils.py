import os
import subprocess
import pandas as pd
import json

import os
from dotenv import load_dotenv
load_dotenv()
REPO_PATH= os.getenv('REPO_PATH')

def combload_topic_dfs(
        topics: list[str] | tuple[str],
        url_function: callable,
        include_topic: bool = False
    ) -> pd.DataFrame:
    """
    Combines and loads the topic DataFrames from the given topics.

    Parameters
    ----------
        topics: list[str] | tuple[str]
            The topics to load.
        url_function: callable
            The function to get the url of the topic.

    Returns
    -------
        pd.DataFrame
    """
    file_type = url_function(topics[0]).split('.')[-1]
    df_list = []

    for topic in topics:
        if file_type == 'csv':
            topic_df = pd.read_csv(
                url_function(topic),
                index_col=0
            )
        elif file_type == 'json':
            topic_df = pd.read_json(
                url_function(topic),
                lines=True,
                orient='records'
            )
        else:
            raise ValueError('File type not supported.')

        if include_topic:
            with open(f'{REPO_PATH}data/topic_data/{topic}_TOPICS.json', 'r') as file:
                topic_dict = json.load(file)

            topic_df['LDA_topic'] = topic_df['storyId'].map(topic_dict)

        topic_df['topic'] = topic
        df_list.append(topic_df)

    df = pd.concat(df_list)

    if include_topic:
        with open(f'{REPO_PATH}data/topic_data/CROSS_TOPICS.json', 'r') as file:
            crosstopic_dict = json.load(file)

        df['cross_topic'] = df['storyId'].map(crosstopic_dict)

    return df





def save_path(
        relative_path: str,
        filename: str
    ) -> str:
    '''
    Returns the absolute path to save a file in the repository.

    Parameters
    ----------
        relative_path (str): The relative path to the file.
        filename (str): The name of the file.

    Returns
    -------
        str: The absolute path to save the file.
    '''
    repo_root = subprocess.check_output("git rev-parse --show-toplevel", shell=True).decode('utf-8').strip()
    save_path = os.path.join(repo_root, relative_path, filename)

    return save_path


def load_df(
        data_folder: str,
        filename: str
    ) -> pd.DataFrame:
    '''
    Loads a news DataFrame from the repository and setting index to datetime.

    Parameters
    ----------
        path (str): The relative path to the news DataFrame.

    Returns
    -------
        pd.DataFrame: The news DataFrame.
    '''
    df = pd.read_csv(fr'C:\Users\joneh\master_thesis\data\{data_folder}\{filename}')
    df.index = pd.to_datetime(df['datetime']).dt.tz_localize(None)
    df.drop(columns=['datetime'], inplace=True)

    return df

def load_json(file_path: str) -> dict[str, str]:
    '''Loads a JSON file as dictionary.

    Parameters
    ----------
        file_path: str
            The path to the JSON file.

    Returns
    -------
        dictionary: Dict[str, str]
            The JSON file as dictionary.
    '''
    with open(file_path, 'r') as file:
        dictionary: dict[str, str] = json.load(file)

    return dictionary
