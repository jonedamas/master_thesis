import pandas as pd
from sqlalchemy import create_engine


def news_db_commit(df: pd.DataFrame):

    engine = create_engine("sqlite:///C:/Users/joneh/master_thesis/data/database/news.db", echo=False)

    df.to_sql('news', con=engine, if_exists='append', index=False)
