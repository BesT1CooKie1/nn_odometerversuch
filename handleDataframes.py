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
    else:
        # Create an empty DataFrame if the file does not exist
        existing_df = pd.DataFrame()

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
                    flattened_entry[f"{sub_key}"] = sub_value
            else:
                flattened_entry[key] = value
        flattened_data.append(flattened_entry)

    # Create a DataFrame from the flattened data
    new_df = pd.DataFrame(flattened_data)

    # Add a row indicating whether the data is input or output
    input_output_row = {key: "Input" if "input" in key.lower() else "Output" for key in new_df.columns}
    input_output_df = pd.DataFrame([input_output_row])

    # Concatenate the input/output row, column headers, and new data
    updated_df = pd.concat([input_output_df, new_df], ignore_index=True)

    if overwrite:
        # Overwrite the existing file
        updated_df.to_excel(file_path, index=False)
        print(f"Data saved to {file_path}")
    else:
        # Append the new data to the existing file
        pd.concat([existing_df, updated_df], ignore_index=True).to_excel(file_path, index=False)

def save_dicts_to_csv(data, file_path, overwrite=False):
    """
    Save a dictionary or a list of dictionaries to a CSV file.

    Parameters
    ----------
    data : dict or list of dicts
        The data to be saved. If a single dictionary is provided, it will be appended to the file.
        If a list of dictionaries is provided, each dictionary will be appended to the file.
    file_path : str
        The path to the CSV file.
    overwrite : bool, optional
        Whether to overwrite the existing file or append the new data. Default is False.
    """
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
                    flattened_entry[f"{sub_key}"] = sub_value
            else:
                flattened_entry[key] = value
        flattened_data.append(flattened_entry)

    # Create a DataFrame from the flattened data
    new_df = pd.DataFrame(flattened_data)

    if overwrite or not os.path.exists(file_path):
        # Overwrite the existing file or create a new one
        new_df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
    else:
        # Append the new data to the existing file
        new_df.to_csv(file_path, mode='a', header=False, index=False)


def save_dicts_to_hdf5(data, file_path, key='data', overwrite=False):
    """
    Save a dictionary or a list of dictionaries to an HDF5 file.

    Parameters
    ----------
    data : dict or list of dicts
        The data to be saved. If a single dictionary is provided, it will be appended to the file.
        If a list of dictionaries is provided, each dictionary will be appended to the file.
    file_path : str
        The path to the HDF5 file.
    key : str, optional
        The key under which the data will be stored in the HDF5 file. Default is 'data'.
    overwrite : bool, optional
        Whether to overwrite the existing file or append the new data. Default is False.
    """
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
                    flattened_entry[f"{sub_key}"] = sub_value
            else:
                flattened_entry[key] = value
        flattened_data.append(flattened_entry)

    # Create a DataFrame from the flattened data
    new_df = pd.DataFrame(flattened_data)

    if overwrite:
        # Overwrite the existing file
        new_df.to_hdf(file_path, key=key, mode='w')
        print(f"Data saved to {file_path}")
    else:
        # Append the new data to the existing file
        new_df.to_hdf(file_path, key=key, mode='a', append=True)