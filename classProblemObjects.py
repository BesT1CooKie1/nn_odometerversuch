# classProblemObjects.py
# Description: This module contains classes for problem objects.

from configparser import ConfigParser
# Import random module
from random import randint

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

    def __init__(self, e_0:float, C_c:float, delta_epsilon:list, initial_stress:list):
        """
        Constructs all the necessary attributes for the Soil object.

        Parameters
        ----------
        e_0 : float
            Initial void ratio
        C_c : float
            Compression index
        delta_epsilon : float
            Strain increment
        """
        units = {
            "Compression Index (C_c)": "[-]",
            "Initial Void Ratio (e_0)": "[-]",
            "Initial Stress (sigma_0)": "[kN/m²]",
            "Additional Stress (delta_sigma)": "[kN/m²]",
            "Strain Increment (delta_epsilon)": "[-]",
            "Effective Stress (sigma_eff)": "[kN/m²]",
            "Shear modulus (E_s": "[N/mm²]"
        }

        self.compression_index = C_c
        self.initial_void_ratio = e_0
        self.strain_increment = delta_epsilon

        self.initial_stress = initial_stress
        self.shear_module = self.__generate_e_s()
        self.additional_stress = self.__generate_additional_stress()
        self.__generate_additional_stress()
        super().__init__(units)
        Soil.instances.append(self)

    def __generate_e_s(self):
        module = []
        for sigma_0 in self.initial_stress:
            E_s = self.__calc_shear_module_e_s(sigma_0)
            module.append(E_s)
        return module

    def __generate_additional_stress(self):
        stress = []
        for E_s in self.shear_module:
            delta_sigma = self.__calc_additional_stress(E_s)
            stress.append(delta_sigma)
        return stress

    def __calc_shear_module_e_s(self, sigma_0):
        """
        Calculate the shear modulus (E_s) in N/mm².

        Returns
        -------
        float
            Shear modulus (E_s) in N/mm².
        """
        return (1 + self.initial_void_ratio) / self.compression_index * sigma_0

    def __calc_additional_stress(self, E_s):
        """
        Calculate the additional stress (delta_sigma) in kN/m².

        Returns
        -------
        float
            Additional stress (delta_sigma) in kN/m².
        """
        return self.strain_increment * E_s

    def __dict__(self):
        return {
            "Compression Index (C_c)": self.compression_index,
            "Initial Void Ratio (e_0)": self.initial_void_ratio,
            "Initial Stress (sigma_0)": self.initial_stress,
            "Additional Stress (delta_sigma)": self.additional_stress,
            "Strain Increment (delta_epsilon)": self.strain_increment,
            "Shear modulus (E_s)": self.shear_module
        }