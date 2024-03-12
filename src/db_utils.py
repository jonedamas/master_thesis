import pandas as pd
from sqlalchemy import create_engine


def news_db_commit(df: pd.DataFrame):
    engine = create_engine("sqlite:///C:/Users/joneh/master_thesis/data/database/news.db", echo=False)
    existing_ids = pd.read_sql_query("SELECT article_id FROM news", engine)['article_id']
    new_entries = df[~df.article_id.isin(existing_ids)]
    if not new_entries.empty:
        new_entries.to_sql('news', con=engine, if_exists='append', index=False)
        print(f"{len(new_entries)} new entries added to the database.")

    else:
        print("No new entries to add to the database.")
