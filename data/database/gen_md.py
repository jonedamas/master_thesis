import datetime

def update_markdown():
    # Define the Markdown file path
    file_path = 'db_status.md'

    # Define the content to add - here, we're just adding a timestamp
    content = f"Last updated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

    # Open the file in append mode ('a') so we add to it without overwriting existing content
    with open(file_path, 'a') as file:
        file.write(content + "\n")  # Write the content with a newline for spacing

if __name__ == "__main__":
    update_markdown()
    print("Markdown file updated.")
