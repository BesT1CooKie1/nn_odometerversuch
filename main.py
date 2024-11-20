import numpy as np
from handleTestValues import generate_test_values
from matplotlib import pyplot as plt

class Boden():
    """
    A class to represent soil properties and calculations.

    Attributes
    ----------
    Cc : float
        Compression index
    Cs : float
        Swelling index
    e0 : float
        Initial void ratio
    sigma0 : float
        Initial stress
    delta_sigma : float
        Change in stress
    sigma_mittel : float
        Mean stress
    E_s_belastung : float
        Stiffness modulus for loading
    E_s_entlastung : float
        Stiffness modulus for unloading
    e_belastung : float
        Void ratio for loading
    e_entlastung : float
        Void ratio for unloading
    """
    def __init__(self, Cc: float, Cs: float, e0: float, sigma0: float, delta_sigma: float):
        """
        Constructs all the necessary attributes for the Boden object.

        Parameters
        ----------
        Cc : float
            Compression index
        Cs : float
            Swelling index
        e0 : float
            Initial void ratio
        sigma0 : float
            Initial stress
        delta_sigma : float
            Change in stress
        """
        self.Cc = Cc
        self.Cs = Cs
        self.e0 = e0
        self.sigma0 = sigma0
        self.delta_sigma = delta_sigma
        self.sigma_mittel = self.sigma0 + self.delta_sigma / 2
        self.E_s_belastung = self.steifenmodul_belastung()
        self.E_s_entlastung = self.steifenmodul_entlastung()
        self.e_belastung = self.porenzahl_belastung(self.sigma_mittel)
        self.e_entlastung = self.porenzahl_entlastung(self.sigma_mittel)

    def steifenmodul_belastung(self):
        """
        Calculate the stiffness modulus for loading considering mean stress.

        Returns
        -------
        float
            Stiffness modulus for loading
        """
        # Berechnung des Steifemoduls für Belastung unter Berücksichtigung von sigma_mittel
        return (1 + self.e0) / self.Cc * self.sigma_mittel

    def steifenmodul_entlastung(self):
        """
        Calculate the stiffness modulus for unloading considering mean stress.

        Returns
        -------
        float
            Stiffness modulus for unloading
        """
        # Berechnung des Steifemoduls für Entlastung unter Berücksichtigung von sigma_mittel
        return (1 + self.e0) / self.Cs * self.sigma_mittel

    # Berechnung der Porenzahl (für Belastung und Entlastung)
    def porenzahl_belastung(self, sigma):
        """
        Calculate the void ratio for loading.

        Parameters
        ----------
        sigma : float
            Stress value

        Returns
        -------
        float
            Void ratio for loading
        """
        return self.e0 - self.Cc * np.log(sigma / self.sigma0)

    def porenzahl_entlastung(self, sigma):
        """
        Calculate the void ratio for unloading.

        Parameters
        ----------
        sigma : float
            Stress value

        Returns
        -------
        float
            Void ratio for unloading
        """
        return self.e0 - self.Cs * np.log(sigma / self.sigma0)

    # Return Funktion zur Ausgabe der Ergebnisse und Inputparameter der Classe
    def print_result(self):
        """
        Print the input parameters and calculated results.
        """
        print("# Materialparameter:")
        print("     Cc:", round(self.Cc, 2), "[-]")
        print("     Cs:", round(self.Cs, 2), "[-]")
        print("     e0:", round(self.e0, 2), "[-]")
        print("     sigma0:", round(self.sigma0, 2), "[kN/m²]")
        print("     delta_sigma:", round(self.delta_sigma, 2), "[kN/m²]")
        print("# Berechnete Ergebnisse:")
        print("     Mittlere Spannung im Schichtmittelpunkt (sigma_mittel):", round(self.sigma_mittel, 2), "[kN/m²]")
        print("     Steifemodul (Belastung):", round(self.E_s_belastung, 2), "[kN/m²]")
        print("     Steifemodul (Entlastung):", round(self.E_s_entlastung, 2), "[kN/m²]")
        print("     Porenzahl (Belastung):", round(self.e_belastung, 2), "[-]")
        print("     Porenzahl (Entlastung):", round(self.e_entlastung, 2), "[-]")

if __name__ == "__main__":
    # Darstellung von 5 Testwertsätzen
    for i in range(5):
        test_values = generate_test_values()
        boden = Boden(test_values["Cc"], test_values["Cs"], test_values["e0"], test_values["sigma0"], test_values["delta_sigma"])
        print(f"## Kennwerte für Boden {i+1}")
        boden.print_result()
        print("\n")
    # Darstellung der Porenzahl in Abhängigkeit von der Spannung

