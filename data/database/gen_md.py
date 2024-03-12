from sqlalchemy import create_engine
from IPython.display import display, Markdown
import pandas as pd
from tabulate import tabulate

def update_markdown():
    # Define the Markdown file path
    file_path = 'data/database/db_status.md'

    engine = create_engine("sqlite:///C:/Users/joneh/master_thesis/data/database/news.db", echo=False)

    # get number of rows in the news table
    query = "SELECT COUNT(*) FROM news"
    num_rows = pd.read_sql(query, engine).iloc[0, 0]

    # get the first 5 rows of the news table
    query = "SELECT * FROM news LIMIT 5"
    df = pd.read_sql(query, engine)

    markdown_table = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)

    # Open the file in write mode ('w') to overwrite existing content
    with open(file_path, 'w') as file:
        file.write(f'## Number of rows in News df: **{num_rows}**')
        file.write('\n\n')
        file.write(markdown_table)

if __name__ == "__main__":
    update_markdown()
    print("Markdown file overwritten.")
