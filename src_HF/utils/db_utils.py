import pandas as pd
from sqlalchemy import create_engine
from IPython.display import Markdown
from tabulate import tabulate

db_description = {
    'news': 'Unaltered news articles, gathered by news article APIs.',
    'news_filtered': 'Filtered news articles through NaN removal, clustering and duplicate removal.',
}


def news_db_commit(df: pd.DataFrame, table: str) -> None:
    '''Commits new entries to the news database.

    Parameters
    ----------
        df: pd.DataFrame
            The DataFrame with new entries to commit to the database.

        table: str
            The table in the database to commit the new entries to.

    Returns
    -------
        None
    '''
    engine = create_engine("sqlite:///C:/Users/joneh/master_thesis/data/database/news.db", echo=False)

    query = f"SELECT article_id FROM {table}"

    existing_ids = pd.read_sql_query(query, engine)['article_id'] # Get existing article ids in the database

    new_entries = df[~df.article_id.isin(existing_ids)] # Get new entries

    if not new_entries.empty:
        new_entries.to_sql(table, con=engine, if_exists='append', index=False)

        print(f"{len(new_entries)} new entries added to the database.")

    else:
        print("No new entries to add to the database.")


def news_db_load(database:str, table: str, dt_index=True) -> pd.DataFrame:
    '''Loads the news database.

    Parameters
    ----------
        database: str
            The database to load.

        table: str
            The table in the database to load.

        dt_index: bool
            Whether to set the datetime column as the index.

    Returns:
        pd.DataFrame: The news database.
    '''
    engine = create_engine(f"sqlite:///C:/Users/joneh/master_thesis/data/database/{database}", echo=False)

    query = f"SELECT * FROM {table}"

    df = pd.read_sql_query(query, engine)

    if dt_index:
        df.index = pd.to_datetime(df['datetime']).dt.tz_localize(None)
        df.drop(columns=['datetime'], inplace=True)

    return df


def db_info(show_table: bool = False) -> Markdown | None:
    '''Writes the database status to a Markdown file.
    '''
    # Define the Markdown file path
    file_path = r'C:\Users\joneh\master_thesis\data\database\README.md'

    engine = create_engine("sqlite:///C:/Users/joneh/master_thesis/data/database/news.db", echo=False)

    output_string = f'# Database Status\n\n'

    # Open the file in write mode ('w') to overwrite existing content
    with open(file_path, 'w') as file:
        # Loop though all tables in the database
        for table in engine.table_names():
            # Write the table name as a header

            output_string += f'## {table}\n\n'

            output_string += f'{db_description[table]}\n\n'

            # get the number of rows in the table
            query = f"SELECT COUNT(*) FROM {table}"
            num_rows = pd.read_sql(query, engine).iloc[0, 0]
            output_string += f'Number of rows: **{num_rows}**.\n\n'

            # unique source tags
            query = f"SELECT source FROM {table}"
            sources = pd.read_sql(query, engine)['source'].unique()
            output_string += f'Unique sources: **{", ".join(sources)}**.\n\n'

             # unique query tags
            query = f"SELECT query FROM {table}"
            sources = pd.read_sql(query, engine)['query'].unique()
            output_string += f'Unique queries: **{", ".join(sources)}**.\n\n'

            if show_table:
                # get the first 3 rows of the table
                query = f"SELECT * FROM {table} LIMIT 3"
                df = pd.read_sql(query, engine)
                markdown_table = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)
                output_string += markdown_table + '\n\n'

        file.write(output_string)

        if __name__ != "__main__":
            return Markdown(output_string)
        else:
            return None

if __name__ == "__main__":
    db_info(show_table=True)
    print("Markdown file overwritten.")
