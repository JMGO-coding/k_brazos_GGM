import numpy as np
from typing import List
from algorithms.algorithm import Algorithm
from arms import Arm

class UCB1(Algorithm):
    def __init__(self, arms: List[Arm]):
        """
        Inicializa el algoritmo UCB1.

        :param arms: Lista de brazos disponibles.
        """
        self.arms = arms
        self.n_arms = len(arms)
        self.counts = np.zeros(self.n_arms)  # Número de veces que cada brazo ha sido seleccionado
        self.values = np.zeros(self.n_arms)  # Media de las recompensas obtenidas para cada brazo
    
    def select_arm(self) -> int:
        """
        Selecciona el brazo con el mayor valor UCB1.

        :return: El índice del brazo seleccionado.
        """
        total_pulls = np.sum(self.counts)  # Total de tiradas realizadas
        ucb_values = np.zeros(self.n_arms)

        for i in range(self.n_arms):
            if self.counts[i] == 0:  # Si nunca ha sido seleccionado, asignamos un valor muy alto para exploración
                ucb_values[i] = float('inf')
            else:
                # Fórmula de UCB1
                ucb_values[i] = self.values[i] + np.sqrt((2 * np.log(total_pulls)) / self.counts[i])

        # Selecciona el brazo con el mayor UCB1
        return np.argmax(ucb_values)

    def update(self, arm_index: int, reward: float):
        """
        Actualiza las estimaciones de la media para el brazo seleccionado.

        :param arm_index: El índice del brazo que fue seleccionado.
        :param reward: La recompensa obtenida de ese brazo.
        """
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        # Actualizamos el valor estimado del brazo usando la media incremental
        self.values[arm_index] = (self.values[arm_index] * (n - 1) + reward) / n

    def run(self, rounds: int):
        """
        Ejecuta el algoritmo UCB1 durante el número especificado de rondas.

        :param rounds: Número total de rondas.
        """
        for t in range(rounds):
            arm_index = self.select_arm()  # Seleccionamos el brazo
            reward = self.arms[arm_index].pull()  # Obtenemos la recompensa del brazo seleccionado
            self.update(arm_index, reward)  # Actualizamos la información de ese brazo


