import numpy as np

from algorithms.algorithm import Algorithm

class AdaptiveSoftmax(Algorithm):
    def __init__(self, k: int, tau_0: float, alpha: float):
        """
        Inicializa el algoritmo AdaptiveSoftmax con ajuste dinámico de tau.

        :param k: Número de brazos.
        :param tau_0: Valor inicial del parámetro tau.
        :param alpha: Parámetro de ajuste para la disminución de tau (0.001 - 1 | Exploracion - Explotacion).
        """
        assert tau_0 > 0, "El parámetro tau_0 debe ser mayor que 0."
        assert alpha >= 0, "El parámetro alpha debe ser no negativo."

        super().__init__(k)
        self.tau_0 = tau_0
        self.alpha = alpha
        self.t = 1  # Contador de iteraciones
        self.counts = np.zeros(k)  # Contador de selecciones por brazo
        self.values = np.zeros(k)  # Estimación de recompensa por brazo

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política Softmax con tau adaptativo.
        :return: índice del brazo seleccionado.
        """
        tau = max(self.tau_0 / (1 + self.alpha * self.t), 1e-2)  # Evitar tau demasiado pequeño
        
        exp_values = np.exp(np.clip(self.values / tau, -500, 500))  # Evita desbordamientos
        sum_exp = np.sum(exp_values)
    
        if sum_exp == 0 or np.isnan(sum_exp):  # Si hay problemas, usar distribución uniforme
            prob = np.ones(self.k) / self.k
        else:
            prob = exp_values / sum_exp
    
        # Verifica que la suma de probabilidades sea 1 y no haya NaN
        assert np.isclose(np.sum(prob), 1), f"Las probabilidades no suman 1: {prob}"
        
        return np.random.choice(self.k, p=prob)


    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza los valores estimados del brazo seleccionado y el contador de iteraciones.
        
        :param chosen_arm: Índice del brazo seleccionado.
        :param reward: Recompensa obtenida del brazo seleccionado.
        """
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        self.values[chosen_arm] += (reward - self.values[chosen_arm]) / n
        self.t += 1
