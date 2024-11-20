import numpy as np

# Funktion zur Generierung eines Kompressionsbewerts (Cc) innerhalb eines realistischen Bereichs
def generate_compression_index():
    # Realistischer Bereich für Cc in geotechnischen Anwendungen (typisch zwischen 0.1 und 0.5)
    return np.random.uniform(0.1, 0.5)

# Funktion zur Generierung eines Schwellbewerts (Cs) innerhalb eines realistischen Bereichs
def generate_swelling_index():
    # Realistischer Bereich für Cs (typisch kleiner als Cc, z.B. zwischen 0.01 und 0.1)
    return np.random.uniform(0.01, 0.1)

# Funktion zur Generierung einer Anfangs-Porenzahl (e0)
def generate_initial_porosity():
    # Typische Werte für e0 liegen im Bereich 0.5 bis 1.5, je nach Materialart
    return np.random.uniform(0.5, 1.5)

# Funktion zur Generierung der Vorbelastung (sigma0) in kN/m²
def generate_initial_stress():
    # Typischer Bereich für die effektive Spannung (sigma0), z.B. zwischen 50 und 200 kN/m²
    return np.random.uniform(50, 200)

# Funktion zur Generierung der Zusatzspannung (delta_sigma) in kN/m²
def generate_additional_stress():
    # Typischer Bereich für Zusatzspannungen in geotechnischen Versuchen, z.B. zwischen 10 und 100 kN/m²
    return np.random.uniform(10, 100)

# Hauptfunktion zur Generierung eines Testwertsatzes
def generate_test_values():
    Cc = generate_compression_index()
    Cs = generate_swelling_index()
    e0 = generate_initial_porosity()
    sigma0 = generate_initial_stress()
    delta_sigma = generate_additional_stress()
    # testssd4s
    return {
        "Cc": Cc,
        "Cs": Cs,
        "e0": e0,
        "sigma0": sigma0,
        "delta_sigma": delta_sigma
    }
