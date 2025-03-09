import numpy as np
from typing import List
from algorithms.algorithm import Algorithm

class UCB2(Algorithm):
    def __init__(self, k: int, alpha: float = 1.0):
        """
        Inicializa el algoritmo UCB2.

        :param k: Número de brazos.
        :param alpha: Parámetro que regula la exploración. Valores más grandes aumentan la exploración.
        """
        # Se asume que Algorithm no necesita inicialización con k, se omite super().__init__(k) si no es necesario
        self.k = k
        self.alpha = alpha  # Parámetro para ajustar la exploración
        self.counts = np.zeros(self.k, dtype=int)  # Veces que cada brazo ha sido seleccionado
        self.values = np.zeros(self.k)  # Media de recompensas obtenidas para cada brazo
    
    def select_arm(self) -> int:
        """
        Selecciona el brazo con el mayor valor UCB2.

        :return: El índice del brazo seleccionado.
        """
        total_pulls = np.sum(self.counts)

        # Si no hay selecciones previas, elegimos un brazo aleatorio
        if total_pulls == 0:
            return np.random.randint(self.k)

        ucb_values = np.zeros(self.k)

        for i in range(self.k):
            if self.counts[i] == 0:  
                ucb_values[i] = float('inf')  # Explorar brazos no seleccionados aún
            else:
                # Fórmula corregida de UCB2
                log_term = np.log(max(1, total_pulls) / self.counts[i])  # Evitar log(0)
                ucb_values[i] = self.values[i] + np.sqrt(((1 + self.alpha) * log_term) / (2 * self.counts[i]))

        return int(np.argmax(ucb_values))  # Asegurar que sea un entero simple

    def update(self, arm_index: int, reward: float):
        """
        Actualiza las estimaciones de la media para el brazo seleccionado.

        :param arm_index: El índice del brazo que fue seleccionado.
        :param reward: La recompensa obtenida de ese brazo.
        """
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        # Actualizamos el valor estimado del brazo usando la media incremental
        self.values[arm_index] += (reward - self.values[arm_index]) / n

