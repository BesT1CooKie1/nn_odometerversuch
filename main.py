import numpy as np
from handleTestValues import generate_test_values
import pandas as pd
from handleDataframes import save_dicts_to_xlsx

class Parent():
    def __dict__(self):
        """
        Return a dictionary representation of the Soil object.

        Returns
        -------
        dict
            Dictionary containing input and output values of the Soil object.
        """
        return {
            "Index": self.index,
            **{f"{key}": value for key, value in self.input_values.items()},
            **{f"{key}": value for key, value in self.output_values.items()}
        }

    def print_result(self):
        """
        Print the input parameters and calculated results.
        """
        print("# Input Parameters:")
        for key, value in self.input_values.items():
            print(f"     {key}:", round(value, 2), "[-]" if "Index" in key else "[kN/m²]")
        print("# Output Results:")
        for key, value in self.output_values.items():
            print(f"     {key}:", round(value, 2), "[-]" if "Void Ratio" in key else "[kN/m²]")

# Get __dict__ method from TestObjects class
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
    """

    instances = []
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

if __name__ == "__main__":
    for i in range(2000):
        test_values = generate_test_values()
        soil = Soil(test_values["Cc"], test_values["Cs"], test_values["e0"], test_values["sigma0"], test_values["delta_sigma"])
        print(f"## Soil Properties {i+1}")
        soil.print_result()
        print("\n")
    data = [soil.__dict__() for soil in Soil.instances]
    save_dicts_to_xlsx(data, "soil_properties.xlsx", overwrite=True)
