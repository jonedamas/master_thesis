import pandas as pd
from typing import List

def aggregate_score(df: pd.DataFrame, colnames: List[str], frequency: str = 'D') -> pd.DataFrame:
    """
    Aggregates the sentiment scores of the given columns in the dataframe

    Args:
        df: pd.DataFrame
        colnames: list[str]

    Returns:
        pd.DataFrame
    """
    
    resampled_df = df[colnames].resample(frequency).mean().dropna()

    return  resampled_df