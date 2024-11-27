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

# Accessing the training configuration
trainOnStartup = config.getboolean('Training', 'TrainOnStartup')
saveModel = config.getboolean('Training', 'SaveModel')
modelPath = config['Training']['ModelPath']

def generate_soil_properties(data_path):
    """
    Generate soil properties and save them to a file.

    Args:
        data_path (str): The path where the soil properties data will be saved.
    """
    start_time = time.time()
    for i in tqdm(range(numberOfTestEntrys), desc="Generating soil properties"):
        test_values = generate_test_values()
        soil = Soil(compression_index=test_values["Cc"], swelling_index=test_values["Cs"],
                    initial_void_ratio=test_values["e0"], initial_stress=test_values["sigma0"],
                    strain_increment=test_values["delta_epsilon"])
    data = [soil.__dict__() for soil in Soil.instances]
    # Save the data to a file
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    save_dicts(data, f"{data_path}soil_properties.{datasetFormat}", file_format=datasetFormat, overwrite=True)

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

def run_oedometer_test():
    """
    Run the oedometer test by generating soil properties and running the neural network.

    This function checks for existing soil properties data and generates new data if necessary.
    It also runs the neural network for the oedometer test.
    """
    file_path = f"{data_path}soil_properties.{datasetFormat}"
    data = load_data(file_path)
    if not os.path.exists(file_path):
        print("Generating new soil properties because the no test-entries exist...")
        generate_soil_properties(data_path)
    elif numberOfTestEntrys != len(data):
        print("Generating new soil properties because the number of test-entries do not match...")
        generate_soil_properties(data_path)
    elif newDatasetOnStartup:
        print("Generating new soil properties because the newTestFilesOnStartup is set to True...")
        generate_soil_properties(data_path)

    if not os.path.exists(modelPath):
        print("Starting the neural network process...")
        input_columns = ['Compression Index (Cc)', 'Swelling Index (Cs)', 'Initial Stress (sigma0)',
                         'Strain Increment (delta_epsilon)']
        output_columns = ['Additional Stress (delta_sigma)']
        run_neural_network(file_path, input_columns, output_columns, mode="OedometerTest")
    elif trainOnStartup:
        print("Starting the neural network process and overwriting the existing model (trainOnStartup is True)...")
        input_columns = ['Compression Index (Cc)', 'Swelling Index (Cs)', 'Initial Stress (sigma0)',
                         'Strain Increment (delta_epsilon)']
        output_columns = ['Additional Stress (delta_sigma)']
        run_neural_network(file_path, input_columns, output_columns, mode="OedometerTest")

if __name__ == "__main__":
    run_oedometer_test()