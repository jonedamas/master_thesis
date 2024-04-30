import pandas as pd
import numpy as np

import os
from dotenv import load_dotenv

load_dotenv()
REPO_PATH = os.getenv('REPO_PATH')



class FuturesBank:
    def __init__(self, init_list: list[str] | None = None):

        self.LCOc1 = pd.read_csv(
            rf'{REPO_PATH}data\time_series\LCOc1_High_Frequency.csv',
            index_col=0
        )
        self.LCOc1.index = pd.to_datetime(self.LCOc1.index)
        self.CLc1 = pd.read_csv(
            rf'{REPO_PATH}data\time_series\CLc1_High_Frequency.csv',
            index_col=0
        )
        self.CLc1.index = pd.to_datetime(self.CLc1.index)

        self.df_list = [self.LCOc1, self.CLc1]

        # execute the init functions
        if init_list:
            for func in init_list:
                getattr(self, func)()

    def add_logret(self) -> None:
        for df in self.df_list:
            df['LOGRET'] = np.log(df['CLOSE']).diff()

    def add_time_features(self) -> None:
        for df in self.df_list:
            df['HOUR'] = df.index.hour
            df['MINUTE'] = df.index.minute
            df['MINUTE_OF_DAY'] = df['HOUR']*60 + df['MINUTE']
            df['DAY_OF_WEEK'] = df.index.dayofweek
            df['DAY_OF_MONTH'] = df.index.day

    def resample_realized(self, df_name: str, frequency: str = '5min'):
        pass
