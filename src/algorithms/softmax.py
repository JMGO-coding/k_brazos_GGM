"""
Module: algorithms/softmax.py
Description: Implementación del algoritmo Softmax para el problema de los k-brazos.

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np

from algorithms.algorithm import Algorithm

class Softmax(Algorithm):

    def __init__(self, k: int, tau: float):
        """
        Inicializa el algoritmo Softmax.

        :param k: Número de brazos.
        :param tau: Parámetro que controla el equilibrio entre exploración y explotación.
        :raises ValueError: Si tau no es mayor que 0.
        """
        assert tau > 0, "El parámetro tau debe ser mayor que 0."

        super().__init__(k)
        self.tau = tau
        self.counts = np.zeros(k)  # Contador de selecciones por brazo
        self.values = np.zeros(k)  # Estimación de recompensa por brazo

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política Softmax.
        :return: índice del brazo seleccionado.
        """

        # Calcular las probabilidades de selección para cada brazo
        exp_values = np.exp(self.values / self.tau)
        prob = exp_values / np.sum(exp_values)

        # Seleccionar un brazo de acuerdo con la distribución de probabilidad calculada
        chosen_arm = np.random.choice(self.k, p=prob)

        return chosen_arm

    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza los valores estimados del brazo seleccionado.

        :param chosen_arm: Índice del brazo seleccionado.
        :param reward: Recompensa obtenida del brazo seleccionado.
        """
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        self.values[chosen_arm] += (reward - self.values[chosen_arm]) / n
