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

class GradienteDePreferencias(Algorithm):
    def __init__(self, k: int, alpha: float = 0.1):
        """
        Inicializa el algoritmo de Gradiente de Preferencias para el problema del bandido de k brazos.

        :param k: Número de brazos.
        :param alpha: Tasa de aprendizaje para actualizar las preferencias.
        """
        super().__init__(k)
        self.alpha = alpha                            # Tasa de aprendizaje
        self.preferences = np.zeros(k)                # Preferencias iniciales de cada brazo inicializadas a 1 par cada brazo
        self.action_probabilities = np.ones(k) / k    # Probabilidades iniciales uniformes
        self.average_reward = 0                       # Promedio de las recompensas recibidas
        self.time_step = 0                            # Contador de iteraciones

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la distribución de probabilidad Softmax sobre las preferencias.
        :return: índice del brazo seleccionado.
        """
        exp_preferences = np.exp(self.preferences - np.max(self.preferences))   # Evita desfases numéricos
        self.action_probabilities = exp_preferences / np.sum(exp_preferences)   # Softmax sobre las preferencias
        return np.random.choice(self.k, p=self.action_probabilities)            # Selección de acciones (brazos) equiprobable

    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza las preferencias del algoritmo basándose en la recompensa obtenida.

        :param chosen_arm: Índice del brazo seleccionado.
        :param reward: Recompensa recibida.
        """
        self.time_step += 1
        self.average_reward += (reward - self.average_reward) / self.time_step  # Actualización incremental de la recompensa promedio
        
        # Actualización de preferencias de cada brazo según el gradiente de preferencias
        baseline = self.average_reward
        for arm in range(self.k):
            if arm == chosen_arm:
                self.preferences[arm] += self.alpha * (reward - baseline) * (1 - self.action_probabilities[arm])
            else:
                self.preferences[arm] -= self.alpha * (reward - baseline) * self.action_probabilities[arm]

