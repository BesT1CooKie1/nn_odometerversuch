import configparser
import numpy as np
import time
from tqdm import tqdm
from handleTestValues import generate_test_values
from handleDataframes import save_dicts_to_xlsx, save_dicts_to_csv, save_dicts_to_hdf5

# Laden der Konfigurationsdatei
config = configparser.ConfigParser()
config.read('config.ini')

# Zugriff auf Konfigurationswerte
data_path = config['DEFAULT']['DataPath']
debug_mode = config.getboolean('DEFAULT', 'Debug')
newTestFilesOnStartup = config.getboolean('DEFAULT', 'NewTestFilesOnStartup')

class Parent:
    """
    Parent class for Soil class
    """

    instances = []
    units = {}

    def __init__(self, units):
        """
        Initialize the Parent class and set units.

        Parameters
        ----------
        units : dict
            Dictionary containing units for the input and output values.
        """
        self.set_units(units)

    def __dict__(self):
        """
        Return a dictionary representation of the Parent object.

        Returns
        -------
        dict
            Dictionary containing input and output values of the Parent object.
        """
        return {
            "Index": self.index,
            **{f"Input - {key}": value for key, value in self.input_values.items()},
            **{f"Output - {key}": value for key, value in self.output_values.items()}
        }

    def __str__(self):
        """
        Return a string representation of the input parameters and calculated results.
        """
        result = "# Input Parameters:\n"
        for key, value in self.input_values.items():
            result += f"     {key}: {round(value, 2)} {self.units.get(key, '')}\n"
        result += "# Output Results:\n"
        for key, value in self.output_values.items():
            result += f"     {key}: {round(value, 2)} {self.units.get(key, '')}\n"
        return result

    def set_units(self, units):
        """
        Set the units for the input and output values.

        Parameters
        ----------
        units : dict
            Dictionary containing units for the input and output values.
        """
        self.units = units

class Soil(Parent):
    """
    A class to represent soil properties and calculations.

    Attributes
    ----------
    compression_index : float
        Compression index
    swelling_index : float
        Swelling index
    initial_void_ratio : float
        Initial void ratio
    initial_stress : float
        Initial stress
    additional_stress : float
        Change in stress
    mean_stress : float
        Mean stress
    stiffness_modulus_loading : float
        Stiffness modulus for loading
    stiffness_modulus_unloading : float
        Stiffness modulus for unloading
    void_ratio_loading : float
        Void ratio for loading
    void_ratio_unloading : float
        Void ratio for unloading
    input_values: dict
    output_values: dict
    units: dict
    """

    def __init__(self, compression_index: float, swelling_index: float, initial_void_ratio: float, initial_stress: float, additional_stress: float):
        """
        Constructs all the necessary attributes for the Soil object.

        Parameters
        ----------
        compression_index : float
            Compression index
        swelling_index : float
            Swelling index
        initial_void_ratio : float
            Initial void ratio
        initial_stress : float
            Initial stress
        additional_stress : float
            Change in stress
        """
        units = {
            "Compression Index": "[-]",
            "Swelling Index": "[-]",
            "Initial Void Ratio": "[-]",
            "Initial Stress": "[kN/m²]",
            "Additional Stress": "[kN/m²]",
            "Stiffness Modulus (Loading)": "[kN/m²]",
            "Stiffness Modulus (Unloading)": "[kN/m²]",
            "Void Ratio (Loading)": "[-]",
            "Void Ratio (Unloading)": "[-]"
        }
        super().__init__(units)
        self.index = len(Soil.instances) + 1
        self.compression_index = compression_index
        self.swelling_index = swelling_index
        self.initial_void_ratio = initial_void_ratio
        self.initial_stress = initial_stress
        self.additional_stress = additional_stress

        self.input_values = {
            "Compression Index": compression_index,
            "Swelling Index": swelling_index,
            "Initial Void Ratio": initial_void_ratio,
            "Initial Stress": initial_stress,
            "Additional Stress": additional_stress
        }
        self.output_values = {}
        self.calculate_all_properties()
        Soil.instances.append(self)

    def calculate_all_properties(self):
        self.mean_stress = self.initial_stress + self.additional_stress / 2
        self.stiffness_modulus_loading = self.calculate_stiffness_modulus_loading()
        self.stiffness_modulus_unloading = self.calculate_stiffness_modulus_unloading()
        self.void_ratio_loading = self.calculate_void_ratio_loading(self.mean_stress)
        self.void_ratio_unloading = self.calculate_void_ratio_unloading(self.mean_stress)

    def calculate_stiffness_modulus_loading(self):
        value = (1 + self.initial_void_ratio) / self.compression_index * self.mean_stress
        self.output_values["Stiffness Modulus (Loading)"] = value
        return value

    def calculate_stiffness_modulus_unloading(self):
        value = (1 + self.initial_void_ratio) / self.swelling_index * self.mean_stress
        self.output_values["Stiffness Modulus (Unloading)"] = value
        return value

    def calculate_void_ratio_loading(self, stress):
        """
        Calculate the void ratio for loading.

        Parameters
        ----------
        stress : float
            Stress value

        Returns
        -------
        float
            Void ratio for loading
        """
        value = self.initial_void_ratio - self.compression_index * np.log(stress / self.initial_stress)
        self.output_values["Void Ratio (Loading)"] = value
        return value

    def calculate_void_ratio_unloading(self, stress):
        """
        Calculate the void ratio for unloading.

        Parameters
        ----------
        stress : float
            Stress value

        Returns
        -------
        float
            Void ratio for unloading
        """
        value = self.initial_void_ratio - self.swelling_index * np.log(stress / self.initial_stress)
        self.output_values["Void Ratio (Unloading)"] = value
        return value


from handleNeuralNetwork import run_neural_network

if __name__ == "__main__":
    if newTestFilesOnStartup:
        # Generate soil properties and save to HDF5 file
        start_time = time.time()
        for i in tqdm(range(1000000), desc="Generating soil properties"):
            test_values = generate_test_values()
            soil = Soil(test_values["Cc"], test_values["Cs"], test_values["e0"], test_values["sigma0"], test_values["delta_sigma"])
            if debug_mode:
                print(f"## Soil Properties {i+1}")
                print(soil)
                print("\n")
        data = [soil.__dict__() for soil in Soil.instances]
        save_dicts_to_hdf5(data, f"{data_path}soil_properties.h5", key='data', overwrite=True)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")
    else:
        print("Data already exists. Skipping generation.")

    # Run the neural network process
    file_path = f"{data_path}soil_properties.h5"
    input_columns = ['Compression Index', 'Swelling Index', 'Initial Void Ratio', 'Initial Stress', 'Additional Stress']
    output_columns = ['Stiffness Modulus (Loading)', 'Stiffness Modulus (Unloading)', 'Void Ratio (Loading)', 'Void Ratio (Unloading)']
    for i in range(len(input_columns)):
        input_columns[i] = "Input - " + input_columns[i]
    for i in range(len(output_columns)):
        output_columns[i] = "Output - " + output_columns[i]
    run_neural_network(file_path, input_columns, output_columns)
