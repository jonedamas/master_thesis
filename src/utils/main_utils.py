import pandas as pd
from IPython.display import display, HTML

from typing import List, Tuple, Dict, Callable, Union
import os
import json
import yaml
import os
from dotenv import load_dotenv

load_dotenv()
REPO_PATH= os.getenv('REPO_PATH')


def apply_nb_style() -> None:
    """
    Apply the notebook style to the output.
    """
    display(
        HTML(
        """
        <style>
        .cell-output-ipywidget-background {
            background-color: transparent !important;
        }
        :root {
            --jp-widgets-color: var(--vscode-editor-foreground);
            --jp-widgets-font-size: var(--vscode-editor-font-size);
        }
        </style>
        """
        )
    )


def combload_topic_dfs(
        topics: Union[List[str], Tuple[str]],
        url_function: Callable,
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
            with open(
                f'{REPO_PATH}data/topic_data/{topic}_TOPICS.json', 'r'
                ) as file:
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


def load_variables(file: str='variable_config.yaml') -> Dict[str, str]:
    """
    Load the variable configuration from the given file.

    Parameters
    ----------
        file: str
            The file to load the variable configuration from.

    Returns
    -------
        dict[str, str]
    """
    with open(f'{REPO_PATH}{file}', 'r') as file:
        var_config = yaml.load(file, Loader=yaml.FullLoader)

    return var_config


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


def load_processed(futures: Union[str, list[str]]) -> Dict[str, pd.DataFrame]:
    """
    Load the processed data for the given futures.

    Parameters
    ----------
        futures: str | list[str]
            The futures to load the data for.

    Returns
    -------
        pd.DataFrame | dict[str, pd.DataFrame]
    """
    if isinstance(futures, str):
        futures = [futures]

    df_dict = {
        future: pd.read_csv(
            os.path.join(
                REPO_PATH,
                'data',
                'prepared_data',
                f"{future}_5min_resampled.csv"
            ),
            index_col='date',
            parse_dates=True
        ) for future in futures
    }

    return df_dict
