from sqlalchemy import create_engine
from IPython.display import display, Markdown
import pandas as pd
from tabulate import tabulate

def db_info():
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

            # get the number of rows in the table
            query = f"SELECT COUNT(*) FROM {table}"
            num_rows = pd.read_sql(query, engine).iloc[0, 0]
            output_string += f'Number of rows: **{num_rows}**.\n\n'

            # get the first 3 rows of the table
            query = f"SELECT * FROM {table} LIMIT 3"
            df = pd.read_sql(query, engine)
            markdown_table = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)

            output_string += markdown_table + '\n\n'

        file.write(output_string)

        if __name__ != "__main__":
            return Markdown(output_string)

if __name__ == "__main__":
    db_info()
    print("Markdown file overwritten.")
