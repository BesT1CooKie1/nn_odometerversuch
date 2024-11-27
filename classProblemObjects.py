# classProblemObjects.py
# Description: This module contains classes for problem objects.

from configparser import ConfigParser
import numpy as np

config = ConfigParser()
config.read('./config/config.ini')

class ProblemObject:
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

    def set_units(self, units):
        """
        Set the units for the input and output values.

        Parameters
        ----------
        units : dict
            Dictionary containing units for the input and output values.
        """
        self.units = units


class Soil(ProblemObject):
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
    strain_increment : float
        Strain increment
    delta_sigma : float
        Change in stress
    units: dict
    """

    def __init__(self, compression_index: float, initial_void_ratio: float, initial_stress: float,
                 strain_increment: float, swelling_index: float = 0, additional_stress: float = 0):
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
        strain_increment : float
        """
        units = {
            "Compression Index (Cc)": "[-]",
            "Swelling Index (Cs)": "[-]",
            "Initial Void Ratio (e0)": "[-]",
            "Initial Stress (sigma0)": "[kN/m²]",
            "Additional Stress (delta_sigma)": "[kN/m²]",
            "Stiffness Modulus (Loading)": "[kN/m²]",
            "Stiffness Modulus (Unloading)": "[kN/m²]",
            "Void Ratio (Loading)": "[-]",
            "Void Ratio (Unloading)": "[-]",
            "Strain Increment (delta_epsilon)": "[-]",
            "Delta Sigma (delta_sigma)": "[kN/m²]",
            "Effective Stress (sigma_eff)": "[kN/m²]"
        }

        super().__init__(units)
        self.index = len(Soil.instances) + 1
        self.compression_index = compression_index
        self.initial_void_ratio = initial_void_ratio
        self.initial_stress = initial_stress
        self.strain_increment = strain_increment
        self.effective_stress = self.calcuclate_effective_stress()
        if additional_stress == 0:
            self.additional_stress = additional_stress
        else:
            self.additional_stress = self.calculate_additional_stress()

        ## Prossible empty
        self.swelling_index = swelling_index
        self.mean_stress = 0
        self.stiffness_modulus_loading = 0
        self.stiffness_modulus_unloading = 0
        self.void_ratio_loading = 0
        self.void_ratio_unloading = 0

        # self.calculate_all_properties()
        Soil.instances.append(self)

    def __dict__(self):
        """
        Return a dictionary representation of the Parent object.

        Returns
        -------
        dict
            Dictionary containing the index and class properties.
        """
        return {
            "Index": self.index,
            "Compression Index (Cc)": self.compression_index,
            "Swelling Index (Cs)": self.swelling_index,
            "Initial Void Ratio (e0)": self.initial_void_ratio,
            "Initial Stress (sigma0)": self.initial_stress,
            "Additional Stress (delta_sigma)": self.additional_stress,
            "Mean Stress (sigma_mean)": self.mean_stress,
            "Stiffness Modulus (Loading)": self.stiffness_modulus_loading,
            "Stiffness Modulus (Unloading)": self.stiffness_modulus_unloading,
            "Void Ratio (Loading)": self.void_ratio_loading,
            "Void Ratio (Unloading)": self.void_ratio_unloading,
            "Strain Increment (delta_epsilon)": self.strain_increment,
            "Effective Stress (sigma_eff)": self.effective_stress
        }

    def __str__(self):
        """
        Return a string representation of the properties
        """
        result = f"Index: {self.index}\n"
        result += f"Compression Index (Cc): {self.compression_index} {self.units['Compression Index (Cc)']}\n" if self.compression_index else ""
        result += f"Swelling Index (Cs): {self.swelling_index} {self.units['Swelling Index (Cs)']}\n" if self.swelling_index else ""
        result += f"Initial Void Ratio (e0): {self.initial_void_ratio} {self.units['Initial Void Ratio (e0)']}\n" if self.initial_void_ratio else ""
        result += f"Initial Stress (sigma0): {self.initial_stress} {self.units['Initial Stress (sigma0)']}\n" if self.initial_stress else ""
        result += f"Additional Stress (delta_sigma): {self.additional_stress} {self.units['Additional Stress (delta_sigma)']}\n" if self.additional_stress else ""
        result += f"Mean Stress (sigma_mean): {self.mean_stress} {self.units['Initial Stress (sigma0)']}\n" if self.mean_stress else ""
        result += f"Stiffness Modulus (Loading): {self.stiffness_modulus_loading} {self.units['Stiffness Modulus (Loading)']}\n" if self.stiffness_modulus_loading else ""
        result += f"Stiffness Modulus (Unloading): {self.stiffness_modulus_unloading} {self.units['Stiffness Modulus (Unloading)']}\n" if self.stiffness_modulus_unloading else ""
        result += f"Void Ratio (Loading): {self.void_ratio_loading} {self.units['Void Ratio (Loading)']}\n" if self.void_ratio_loading else ""
        result += f"Void Ratio (Unloading): {self.void_ratio_unloading} {self.units['Void Ratio (Unloading)']}\n" if self.void_ratio_unloading else ""
        result += f"Strain Increment (delta_epsilon): {self.strain_increment} {self.units['Strain Increment (delta_epsilon)']}\n" if self.strain_increment else ""
        result += f"Effective Stress (sigma_eff): {self.effective_stress} {self.units['Effective Stress (sigma_eff)']}\n" if self.effective_stress else ""
        return result

    def calculate_all_properties(self):
        self.mean_stress = self.initial_stress + self.additional_stress / 2
        self.stiffness_modulus_loading = self.calculate_stiffness_modulus_loading()
        self.stiffness_modulus_unloading = self.calculate_stiffness_modulus_unloading()
        self.void_ratio_loading = self.calculate_void_ratio_loading()
        self.void_ratio_unloading = self.calculate_void_ratio_unloading()
        self.delta_sigma = self.calculate_delta_sigma()


    def calculate_stiffness_modulus_loading(self):
        """
        Calculate the stiffness modulus for loading.

        Returns
        -------
        float
            Stiffness modulus for loading
        """
        value = (1 + self.initial_void_ratio) / self.compression_index * self.mean_stress
        return value


    def calculate_stiffness_modulus_unloading(self):
        """
        Calculate the stiffness modulus for unloading.

        Returns
        -------
        float
            Stiffness modulus for unloading
        """
        value = (1 + self.initial_void_ratio) / self.swelling_index * self.mean_stress
        return value


    def calculate_void_ratio_loading(self):
        """
        Calculate the void ratio for loading.


        Returns
        -------
        float
            Void ratio for loading
        """
        value = self.initial_void_ratio - self.compression_index * np.log(self.mean_stress / self.initial_stress)
        return value


    def calculate_void_ratio_unloading(self):
        """
        Calculate the void ratio for unloading.

        Returns
        -------
        float
            Void ratio for unloading
        """
        value = self.initial_void_ratio - self.swelling_index * np.log(self.mean_stress / self.initial_stress)
        return value


    def calculate_additional_stress(self):
        """
        Calculate the change in stress (Delta Sigma) based on the class attributes.

        Returns
        -------
        float
            Change in stress (Delta Sigma).
        """
        return self.effective_stress - self.initial_stress

    def calcuclate_effective_stress(self):
        """
        Calculate the effective stress based on the class attributes.

        Returns
        -------
        float
            Effective stress.
        """
        return self.initial_stress * 10**(((1+self.initial_void_ratio)*self.strain_increment)/self.compression_index)

    def calculate_pore_count(self):
        """
        Calculate the pore count based on the class attributes.

        Returns
        -------
        int
            Pore count.
        """
        return float(self.initial_void_ratio - (1 + self.initial_void_ratio) * self.strain_increment)
