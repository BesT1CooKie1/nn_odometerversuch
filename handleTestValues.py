# handleTestValues.py
# Description: This module contains functions to generate test values for soil properties.

import numpy as np
from random import randint

def generate_compression_index():
    """
    Generate a compression index (Cc) within a realistic range.

    Returns
    -------
    float
        Compression index (Cc) typically between 0.1 and 0.5 for geotechnical applications.
    """
    return np.random.uniform(0.001, 1)


def generate_swelling_index():
    """
    Generate a swelling index (Cs) within a realistic range.

    Returns
    -------
    float
        Swelling index (Cs) typically smaller than Cc, e.g., between 0.01 and 0.1.
    """
    return np.random.uniform(0.0001, 0.4)


def generate_initial_porosity():
    """
    Generate an initial void ratio (e0).

    Returns
    -------
    float
        Initial void ratio (e0) typically between 0.5 and 1.5 depending on the material type.
    """
    return np.random.uniform(0.5, 1.5)


def generate_initial_stress():
    """
    Generate the initial stress (sigma0) in kN/m².

    Returns
    -------
    float
        Initial stress (sigma0) typically between 50 and 200 kN/m².
    """
    stress = []
    value = randint(50, 200)
    for i in range(100):
        stress.append(round(value, 0))
        value = value + randint(5, 25)
    return stress


def generate_additional_stress():
    """
    Generate the additional stress (delta_sigma) in kN/m².

    Returns
    -------
    float
        Additional stress (delta_sigma) typically between 10 and 100 kN/m².
    """
    return np.random.uniform(10, 100)


def generate_strain_increment():
    """
    Generate a realistic strain increment (delta_epsilon) for geotechnical tests.

    Returns
    -------
    float
        Strain increment (delta_epsilon) typically between 0.0001 and 0.01.
    """
    stress = []
    value = randint(50, 200)
    for i in range(100):
        stress.append(round(value, 0))
        value = value + randint(5, 25)
    return stress

def generate_test_values():
    """
    Generate a set of test values for soil properties.

    Returns
    -------
    dict
        Dictionary containing generated test values for compression index (Cc), swelling index (Cs),
        initial void ratio (e0), initial stress (sigma0), and additional stress (delta_sigma).
    """
    C_c = 0.005
    #C_s = generate_swelling_index()
    e_0 = 1.00
    sigma_0 = generate_initial_stress()
    delta_epsilon = generate_strain_increment()
    return {
        "C_c": C_c,
    #    "Cs": C_s,
        "e_0": e_0,
        "sigma_0": sigma_0,
        "delta_epsilon": delta_epsilon
    }
