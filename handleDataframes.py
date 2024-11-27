# handleDataframes.py
# Description: This module contains functions to handle DataFrames.

import pandas as pd
import os

def save_dicts(data, file_path, file_format, key='data', overwrite=False):
    """
    Save a list of dictionaries to a file (Excel, CSV, or HDF5).

    Parameters
    ----------
    data : list of dicts
        The data to be saved. Each dictionary represents a row in the file.
    file_path : str
        The path to the file.
    file_format : str
        The format of the file ('xlsx', 'csv', 'hdf5').
    key : str, optional
        The key under which the data will be stored in the HDF5 file. Default is 'data'.
    overwrite : bool, optional
        Whether to overwrite the existing file or append the new data. Default is False.
    """

    if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
        raise ValueError("Data must be a list of dictionaries")

    # Create a DataFrame from the data
    new_df = pd.DataFrame(data)

    if file_format == 'xlsx':
        if os.path.exists(file_path) and not overwrite:
            existing_df = pd.read_excel(file_path)
            pd.concat([existing_df, new_df], ignore_index=True).to_excel(file_path, index=False)
        else:
            new_df.to_excel(file_path, index=False)
        print(f"Dataset saved to {file_path}")

    elif file_format == 'csv':
        if overwrite or not os.path.exists(file_path):
            new_df.to_csv(file_path, index=False)
        else:
            new_df.to_csv(file_path, mode='a', header=False, index=False)
        print(f"Dataset saved to {file_path}")

    elif file_format == 'hdf5' or file_format == 'h5':
        if overwrite or not os.path.exists(file_path):
            new_df.to_hdf(file_path, key=key, mode='w')
        else:
            new_df.to_hdf(file_path, key=key, mode='a', append=True)
        print(f"Dataset saved to {file_path}")
    else:
        raise ValueError("Unsupported file format. Please use 'xlsx', 'csv', or 'hdf5'.")


def load_data(file_path):
    """
    Loads data from a file.

    Parameters:
    file_path (str): The path to the data file. Supported formats are .xlsx, .csv, and .h5.

    Returns:
    pd.DataFrame: The loaded data as a pandas DataFrame.

    Raises:
    ValueError: If the file format is not supported.
    """
    if file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.hdf5') or file_path.endswith('.h5'):
        data = pd.read_hdf(file_path, key='data')
    else:
        raise ValueError("Unsupported file format. Please use .xlsx, .csv, or .h5")
    return data

def clear_columns(data, columns):
    """
    Clear the columns from the DataFrame which are not in the given list.

    Parameters
    ----------
    data : DataFrame
        The DataFrame to be cleaned.
    columns : list
        The list of columns to keep in the DataFrame.
    """
    data.drop(data.columns.difference(columns), axis=1, inplace=True)