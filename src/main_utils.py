import os
import subprocess

def save_path(
        relative_path: str, 
        filename: str
    ) -> str:
    '''
    Returns the absolute path to save a file in the repository.

    Args:
        relative_path (str): The relative path to the file.
        filename (str): The name of the file.
    
    Returns:
        str: The absolute path to save the file.
    '''
    repo_root = subprocess.check_output("git rev-parse --show-toplevel", shell=True).decode('utf-8').strip()
    save_path = os.path.join(repo_root, relative_path, filename)

    return save_path


