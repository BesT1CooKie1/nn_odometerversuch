# main.py
# Description: This module contains the main function to run the oedometer test.

import os
import configparser
import time
from tqdm import tqdm
from classProblemObjects import Soil
from handleDataframes import save_dicts, load_data
from handleNeuralNetwork import run_neural_network
from handleTestValues import generate_test_values

# Load the configuration file
config = configparser.ConfigParser()
config.read('./config/config.ini')

# Accessing the initialization configuration
data_path = config['Init']['DataPath']
debug_mode = config.getboolean('Init', 'Debug')
newDatasetOnStartup = config.getboolean('Init', 'newDatasetOnStartup')
numberOfTestEntrys = config.getint('Init', 'SizeOfDataset')
datasetFormat = config['Init']['DatasetFormat']
debugPraefix = config['Init']['DebugPraefix']


# Accessing the training configuration
overwrite = config.getboolean('Training', 'Overwrite')
saveModel = config.getboolean('Training', 'SaveModel')
neuronalNetworkEnable = config.getboolean('Training', 'Enable')
modelPath = config['Training']['ModelPath']

def generate_soil_properties(data_path):
    """
    Generate soil properties and save them to a file.

    Args:
        data_path (str): The path where the soil properties data will be saved.
    """
    start_time = time.time()
    if debug_mode:
        print(f"{debugPraefix}Initalizing the generation of soil properties with {numberOfTestEntrys} test entries...")
    for i in tqdm(range(numberOfTestEntrys), desc="Generating soil properties"):
        test_values = generate_test_values()
        # def __init__(self, e_0:float, C_c:float, delta_epsilon:list, initial_stress:list):
        Soil(test_values['e_0'], test_values['C_c'], test_values['delta_epsilon'], test_values['sigma_0'])
    data = [soil.__dict__() for soil in Soil.instances]
    # Save the data to a file
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    save_dicts(data, f"{data_path}soil_properties.{datasetFormat}", file_format=datasetFormat, overwrite=True)
    if debug_mode:
        print(f"{debugPraefix}Creating an Excel file for the dataset to check the values...\n{debugPraefix}Disabling debug mode will speed up the process and create no Excel file.\n{debugPraefix}Check the config.ini file to turn off the debug mode.")
        save_dicts(data, f"{data_path}soil_properties.xlsx", file_format="xlsx", overwrite=True)

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

def run_oedometer_test():
    """
    Run the oedometer test by generating soil properties and running the neural network.
    """
    file_path = f"{data_path}soil_properties.{datasetFormat}"
    try:
        data = load_data(file_path)
    except FileNotFoundError:
        data = []

    if not os.path.exists(file_path) or numberOfTestEntrys != len(data) or newDatasetOnStartup:
        print("Generating new soil properties...")
        generate_soil_properties(data_path)

    if not os.path.exists(modelPath) or overwrite:
        print("Starting the neural network process...")
        input_columns = ['Initial Stress (sigma_0)', 'Strain Increment (delta_epsilon)', ]
        output_columns = ['Additional Stress (delta_sigma)']
        if neuronalNetworkEnable:
            run_neural_network(file_path, input_columns, output_columns, mode="OedometerTest")

if __name__ == "__main__":
    run_oedometer_test()