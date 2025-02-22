"""
Module: plotting/plotting.py
Description: Contiene funciones para generar gráficas de comparación de algoritmos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

# [1] - Importamos las lobrerías y clases necesarias
from typing import List
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from algorithms import Algorithm, EpsilonGreedy

# [2] - Definimos algunas funciones de visualización
def get_algorithm_label(algo: Algorithm) -> str:
    """
    Genera una etiqueta descriptiva para el algoritmo incluyendo sus parámetros.

    :param algo: Instancia de un algoritmo.
    :type algo: Algorithm
    :return: Cadena descriptiva para el algoritmo.
    :rtype: str
    """
    label = type(algo).__name__
    if isinstance(algo, EpsilonGreedy):
        label += f" (epsilon={algo.epsilon})"
    elif isinstance(algo, UCB1):
         label = label
    elif isinstance(algo, UCB2):
         label += f" (alpha={algo.alpha})"
    elif isinstance(algo, GradienteDePreferencias):
         label += f" (alpha={algo.alpha})"
    elif isinstance(algo, Softmax):
         label += f" (tau={algo.tau})"
    else:
        raise ValueError("El algoritmo debe ser de la clase Algorithm o una subclase.")
    return label


def plot_average_rewards(steps: int, rewards: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Recompensa Promedio vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param rewards: Matriz de recompensas promedio.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), rewards[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Recompensa Promedio', fontsize=14)
    plt.title('Recompensa Promedio vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()


def plot_optimal_selections(steps: int, optimal_selections: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param optimal_selections: Matriz de porcentaje de selecciones óptimas.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """ 

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):                                           # Recorriendo los algoritmos del listado...
        label = get_algorithm_label(algo)                                            
        plt.plot(range(steps), optimal_selections[idx], label=label, linewidth=2)         # ...hacemos un plot del número de elecciones de brazo óptimas acmulativa respecto del número de etapas del algoritmo
    
    # Tuning del gráfico para una mejor visualización 
    plt.xlabel('Pasos de Tiempo', fontsize=14)        
    plt.ylabel('Porcentaje de elecciones óptimas', fontsize=14)
    plt.title('Porcentaje de elecciones óptimas vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()

    # raise NotImplementedError("Esta función aún no ha sido implementada.")


def plot_arm_statistics(arm_stats: List[dict], algorithms: List[Algorithm], optimal_arms_list, num_choices_list, *args):
    """
    Genera gráficas separadas de Selección de Arms:
    Ganancias vs Pérdidas para cada algoritmo.
    :param arm_stats: Lista (de diccionarios) con estadísticas de cada brazo por algoritmo.
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param num_choices_list: Lista de listas que indican el nº de veces que se elige cada brazo del bandido en la ejecución de cada algoritmo
    :param optimal_arms_list: lista que contiene el número de brazo óptimo para cada algoritmo en ``algorithms``
    :param args: Parámetros que consideres
    """
    
    # Definimos internamente una función que crea histogramas de forma algo más genérica para después particularizar con los requierimientos de ``plot_arm_statistics``
    def plot_histograms(n, data_list, configs, highlight_bars, str_choices, text_color="blue", vertical_spacing=1):
        """
        Genera una figura de N x 1 histogramas con configuraciones personalizadas y etiquetas sobre las barras.

        :param n: Número de histogramas
        :param data_list: Lista de conjuntos de datos para cada histograma
        :param configs: Lista de diccionarios con configuraciones para cada histograma
        :param highlight_bars: Lista de listas de índices de barras a resaltar en cada histograma
        :param str_choices: Lista de listas con etiquetas de texto para cada barra
        :param text_color: Color del texto de las etiquetas sobre las barras
        :param vertical_spacing: Espaciado vertical entre los subplots
        """
        fig, axes = plt.subplots(n, 1, figsize=(12, 7 * n), constrained_layout=True)
        fig.set_layout_engine(None)
        fig.subplots_adjust(hspace=vertical_spacing)

        if n == 1:
            axes = [axes]
            
        # Creamos una gráfica para cada histograma y seleccinoamos para cada una, unos datos que graficar y una configuración de plot.
        for i, ax in enumerate(axes):
            data = data_list[i]
            config = configs[i]
            highlight_indices = highlight_bars[i] if i < len(highlight_bars) else []
            choices = str_choices[i] if i < len(str_choices) else []
            color = config.get("color", "gray")
            highlight_color = config.get("highlight_color", "red")
            edgecolor = config.get("edgecolor", "black")
            alpha = config.get("alpha", 0.7)
            
            x_labels = list(range(1, len(data) + 1))
            bars = ax.bar(x_labels, data, color=color, edgecolor=edgecolor, alpha=alpha)

            for j, bar in enumerate(bars):
                if j in highlight_indices:
                    bar.set_color(highlight_color)

                label = choices[j] if j < len(choices) else ""
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, 
                        ha='center', va='bottom', fontsize=10, color=text_color)
            
            ax.set_title(config.get("title", f"Gráfico de Barras {i+1}"))
            ax.set_xlabel(config.get("xlabel", "Brazo seleccionado"))
            ax.set_xticks(x_labels)
            ax.set_ylabel(config.get("ylabel", "Promedio de ganancias por brazo"))
            ax.grid(config.get("grid", True))
        
        plt.show()
        
    # Ahora llamamos a la función de visualización de histogramas pasándole los datos de entrada 
    n = len(algorithms)
    data_list = [[arm_stats[j][key]["reward"] for key in arm_stats[j].keys()] for j in range(len(arm_stats))]
    configs = [{"color": "gray", "bins": len(arm_stats[0].keys()), "title": f"Histograma {get_algorithm_label(alg)}", "highlight_color": "red"} for alg in algorithms]
    highlight_bars = [[num] for num in optimal_arms_list]
    str_choices = [["N"+str(j+1)+": "+str(num_choices_list[i][j]) for j in range(len(num_choices_list[i]))] for i in range(len(num_choices_list))]
    plot_histograms(n, data_list, configs, highlight_bars, str_choices=str_choices)

   
    ''' - COMENTARIO SOBRE ESTRUCTURA DE LOS DATOS EN ``data_list``, BORRAR CUANDO EL CÓDIGO FUNCIONE -

    d1 = {"arm_1" : {"media":mu_1 , "std" : sigma_1},
          "arm_2" : {"media":mu_2 , "std" : sigma_2}}
    
    [d1,d2,d3]
    '''    


def plot_regret(steps: int, regret_accumulated: np.ndarray, algorithms: List[Algorithm], theoretical_bound: np.ndarray = None):
    """
    Genera la gráfica de Regret Acumulado vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param regret_accumulated: Matriz de regret acumulado (algoritmos x pasos).
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param theoretical_bound: (Opcional) Array con la cota teórica del regret Cte * ln(T).
    """
    
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), regret_accumulated[idx], label=label, linewidth=2)

    # Si se proporciona una cota teórica, se grafica la cota como una línea límite
    if theoretical_bound is not None:
        plt.plot(range(steps), theoretical_bound, 'k--', label="Cota teórica", linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Regret Acumulado', fontsize=14)
    plt.title('Regret Acumulado vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()
