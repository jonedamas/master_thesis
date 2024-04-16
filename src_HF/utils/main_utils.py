import os
import subprocess
import pandas as pd
import json
from typing import Dict

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

def load_json(file_path: str) -> Dict[str, str]:
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
        dictionary = json.load(file)

    return dictionary