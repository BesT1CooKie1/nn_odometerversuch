import pandas as pd
import os

import pandas as pd
import os

def save_dicts_to_xlsx(data, file_path, overwrite=False):
    """
    Save a dictionary or a list of dictionaries to an Excel file.

    Parameters
    ----------
    data : dict or list of dicts
        The data to be saved. If a single dictionary is provided, it will be appended to the file.
        If a list of dictionaries is provided, each dictionary will be appended to the file.
    file_path : str
        The path to the Excel file.
    """
    # Check if the file exists
    if os.path.exists(file_path):
        # Load the existing file
        existing_df = pd.read_excel(file_path)
        existing_columns = set(existing_df.columns)
    else:
        # Create an empty DataFrame if the file does not exist
        existing_df = pd.DataFrame()
        existing_columns = set()

    # Ensure data is a list of dictionaries
    if isinstance(data, dict):
        data = [data]

    # Check if all keys in the dictionaries match the existing columns
    for entry in data:
        if not set(entry.keys()).issubset(existing_columns):
            # If the file is new or columns do not match, create a new DataFrame with the data
            existing_df = pd.DataFrame(columns=entry.keys())
            existing_columns = set(entry.keys())
            break

    # Create a DataFrame from the data
    new_df = pd.DataFrame(data)

    if overwrite:
        # Overwrite the existing file
        new_df.to_excel(file_path, index=False)
    else:
        # Append the new data to the existing file
        pd.concat([existing_df, new_df], ignore_index=True).to_excel(file_path, index=False)
