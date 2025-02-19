"""
Module: arms/armBinomial.py
Description: Contains the implementation of the Armbinomial class for the binomial distribution arm.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""


import numpy as np

from arms import Arm


class ArmBinomial(Arm):
    def __init__(self, n: int, p: float):
        """
        Inicializa el brazo con distribución binomial.

        :param n: Número de ensayos.
        :param p: probabilidad de éxito en cada ensayo.
        """
        assert n > 0,       "El número de muestras debe ser positivo."
        assert 0 <= p <= 1, "p ha de ser una probabilidad."

        self.n = int(n)
        self.p = p
        self.mu = self.n*self.p

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución binomial.

        :return: Recompensa obtenida del brazo.
        """
        reward = np.random.binomial(n=self.n, p=self.p)

        return reward

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución binomial.

        :return: Valor esperado de la distribución.
        """

        return self.mu

    def __str__(self):
        """
        Representación en cadena del brazo binomial.

        :return: Descripción detallada del brazo binomial.
        """
        return f"ArmBinomial(n={self.n}, p={self.p})"

    @classmethod
    def generate_arms(cls, k: int, n_min: int = 1, n_max: int = 25):
        """
        Genera k brazos con numero de muestras en el rango [1, 25].

        :param k: Número de brazos a generar.
        :param n_min: Valor mínimo del número de muestras.
        :param n_max: Valor máximo del número de muestras.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert n_min < n_max, "El valor de mu_min debe ser menor que mu_max."

        # Generar k- valores únicos de n
        np_values = set()
        while len(np_values) < k:
            n = int(np.random.uniform(n_min, n_max))
            p = np.random.uniform(0, 1)
            np_values.add((n,p))

        np_values = list(np_values)
        arms = [ArmBinomial(val[0], val[1]) for val in np_values]

        return arms
