"""
Module: algorithms/epsilon_greedy.py
Description: Implementación del algoritmo epsilon-greedy para el problema de los k-brazos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np

from algorithms.algorithm import Algorithm


class EpsilonGreedy(Algorithm):

    def __init__(self, k: int, epsilon: float = 0.1):
        """
        Inicializa el algoritmo epsilon-greedy.

        :param k: Número de brazos.
        :param epsilon: Probabilidad de exploración (seleccionar un brazo al azar).
        :raises ValueError: Si epsilon no está en [0, 1].
        """
        assert 0 <= epsilon <= 1, "El parámetro epsilon debe estar entre 0 y 1."

        super().__init__(k)
        self.epsilon = epsilon
        self.initial_exploration = set(range(k))  # Conjunto de brazos aún no explorados

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política epsilon-greedy, asegurando que cada brazo
        sea seleccionado al menos una vez antes de iniciar la exploración/explotación normal.
        :return: índice del brazo seleccionado.
        """
        if self.initial_exploration:
            # Extraer y eliminar un brazo del conjunto de brazos sin explorar
            return self.initial_exploration.pop()
        
        if np.random.random() < self.epsilon:
            # Selecciona un brazo al azar
            return np.random.choice(self.k)
        else:
            # Selecciona el brazo con la recompensa promedio estimada más alta
            return np.argmax(self.values)    # Chosen arm


