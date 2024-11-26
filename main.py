import numpy as np
from handleTestValues import generate_test_values
import pandas as pd
from handleDataframes import save_dicts_to_xlsx

class Soil():
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
    instances : list
        List of all Soil instances
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
        self.mean_stress = self.initial_stress + self.additional_stress / 2
        self.stiffness_modulus_loading = self.calculate_stiffness_modulus_loading()
        self.stiffness_modulus_unloading = self.calculate_stiffness_modulus_unloading()
        self.void_ratio_loading = self.calculate_void_ratio_loading(self.mean_stress)
        self.void_ratio_unloading = self.calculate_void_ratio_unloading(self.mean_stress)
        Soil.instances.append(self)

    def calculate_stiffness_modulus_loading(self):
        """
        Calculate the stiffness modulus for loading considering mean stress.

        Returns
        -------
        float
            Stiffness modulus for loading
        """
        return (1 + self.initial_void_ratio) / self.compression_index * self.mean_stress

    def calculate_stiffness_modulus_unloading(self):
        """
        Calculate the stiffness modulus for unloading considering mean stress.

        Returns
        -------
        float
            Stiffness modulus for unloading
        """
        return (1 + self.initial_void_ratio) / self.swelling_index * self.mean_stress

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
        return self.initial_void_ratio - self.compression_index * np.log(stress / self.initial_stress)

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
        return self.initial_void_ratio - self.swelling_index * np.log(stress / self.initial_stress)

    def print_result(self):
        """
        Print the input parameters and calculated results.
        """
        print("# Material Parameters:")
        print("     Compression Index:", round(self.compression_index, 2), "[-]")
        print("     Swelling Index:", round(self.swelling_index, 2), "[-]")
        print("     Initial Void Ratio:", round(self.initial_void_ratio, 2), "[-]")
        print("     Initial Stress:", round(self.initial_stress, 2), "[kN/m²]")
        print("     Additional Stress:", round(self.additional_stress, 2), "[kN/m²]")
        print("# Calculated Results:")
        print("     Mean Stress:", round(self.mean_stress, 2), "[kN/m²]")
        print("     Stiffness Modulus (Loading):", round(self.stiffness_modulus_loading, 2), "[kN/m²]")
        print("     Stiffness Modulus (Unloading):", round(self.stiffness_modulus_unloading, 2), "[kN/m²]")
        print("     Void Ratio (Loading):", round(self.void_ratio_loading, 2), "[-]")
        print("     Void Ratio (Unloading):", round(self.void_ratio_unloading, 2), "[-]")

    def __dict__(self):
        """
        Return a dictionary representation of the Soil object.

        Returns
        -------
        dict
            Dictionary containing all attributes of the Soil object.
        """
        return {
            "Index": self.index,
            "Compression Index": self.compression_index,
            "Swelling Index": self.swelling_index,
            "Initial Void Ratio": self.initial_void_ratio,
            "Initial Stress": self.initial_stress,
            "Additional Stress": self.additional_stress,
            "Mean Stress": self.mean_stress,
            "Stiffness Modulus (Loading)": self.stiffness_modulus_loading,
            "Stiffness Modulus (Unloading)": self.stiffness_modulus_unloading,
            "Void Ratio (Loading)": self.void_ratio_loading,
            "Void Ratio (Unloading)": self.void_ratio_unloading
        }

if __name__ == "__main__":
    for i in range(2000):
        test_values = generate_test_values()
        soil = Soil(test_values["Cc"], test_values["Cs"], test_values["e0"], test_values["sigma0"], test_values["delta_sigma"])
        print(f"## Soil Properties {i+1}")
        soil.print_result()
        print("\n")
    data = [soil.__dict__() for soil in Soil.instances]
    save_dicts_to_xlsx(data, "soil_properties.xlsx", overwrite=True)
