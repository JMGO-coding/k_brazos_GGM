"""
Module: arms/armBernoulli.py
Description: Contains the implementation of the ArmBernoulli class for the Bernoulli distribution arm.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""


import numpy as np

from arms import Arm


class ArmBernoulli(Arm):
    def __init__(self, p: float):
        """
        Inicializa el brazo con distribución Bernoulli.

        :param p: Probabilidad de éxito en un solo ensayo.
        """
        assert 0 <= p <= 1, "p ha de ser una probabilidad."

        self.p = p
        self.mu = self.p  # En una Bernoulli, la media es p.

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución Bernoulli.

        :return: Recompensa obtenida del brazo (0 o 1).
        """
        reward = np.random.binomial(n=1, p=self.p)  # Usamos n=1 para Bernoulli

        return reward

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución Bernoulli.

        :return: Valor esperado de la distribución.
        """
        return self.mu

    def __str__(self):
        """
        Representación en cadena del brazo Bernoulli.

        :return: Descripción detallada del brazo Bernoulli.
        """
        return f"ArmBernoulli(p={self.p})"

    @classmethod
    def generate_arms(cls, k: int):
        """
        Genera k brazos con diferentes valores de probabilidad p.

        :param k: Número de brazos a generar.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."

        # Generar k valores únicos de p
        p_values = set()
        while len(p_values) < k:
            p = np.random.uniform(0, 1)
            p_values.add(p)

        p_values = list(p_values)
        arms = [ArmBernoulli(p) for p in p_values]

        return arms

