# handleTestValues.py
# Description: This module contains functions to generate test values for soil properties.

import numpy as np

def generate_compression_index():
    """
    Generate a compression index (Cc) within a realistic range.

    Returns
    -------
    float
        Compression index (Cc) typically between 0.1 and 0.5 for geotechnical applications.
    """
    return np.random.uniform(0.1, 0.5)


def generate_swelling_index():
    """
    Generate a swelling index (Cs) within a realistic range.

    Returns
    -------
    float
        Swelling index (Cs) typically smaller than Cc, e.g., between 0.01 and 0.1.
    """
    return np.random.uniform(0.01, 0.1)


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
    return np.random.uniform(50, 200)


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
    return np.random.uniform(0.0001, 0.01)


def generate_test_values():
    """
    Generate a set of test values for soil properties.

    Returns
    -------
    dict
        Dictionary containing generated test values for compression index (Cc), swelling index (Cs),
        initial void ratio (e0), initial stress (sigma0), and additional stress (delta_sigma).
    """
    Cc = generate_compression_index()
    Cs = generate_swelling_index()
    e0 = generate_initial_porosity()
    sigma0 = generate_initial_stress()
    delta_epsilon = generate_strain_increment()
    return {
        "Cc": Cc,
        "Cs": Cs,
        "e0": e0,
        "sigma0": sigma0,
        "delta_epsilon": delta_epsilon
    }
