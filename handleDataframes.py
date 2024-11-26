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
    overwrite : bool, optional
        Whether to overwrite the existing file or append the new data. Default is False.
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

    # Flatten the dictionaries
    flattened_data = []
    for entry in data:
        flattened_entry = {}
        for key, value in entry.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    print(sub_key)
                    flattened_entry[f"{sub_key}"] = sub_value
            else:
                flattened_entry[key] = value
        flattened_data.append(flattened_entry)

    # Create a DataFrame from the flattened data
    new_df = pd.DataFrame(flattened_data)

    # Add a row indicating whether the data is input or output
    input_output_row = {key: "Input" if "Input" in key else "Output" for key in new_df.columns}
    input_output_df = pd.DataFrame([input_output_row])

    # Concatenate the input/output row, column headers, and new data
    updated_df = pd.concat([input_output_df, new_df], ignore_index=True)

    # Flatten the columns
    updated_df.columns = [f"{col}" for col in updated_df.columns]

    if overwrite:
        # Overwrite the existing file
        updated_df.to_excel(file_path, index=False)
    else:
        # Append the new data to the existing file
        pd.concat([existing_df, updated_df], ignore_index=True).to_excel(file_path, index=False)