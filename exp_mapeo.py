import numpy as np
import matplotlib.pyplot as plt

"""Genera un fractal basado en la función exponencial en el plano complejo."""
def exponential_map_fractal(xmin = -2, xmax = 2, ymin = -2 , ymax = 2, width = 800, height = 800, 
                            c = 5 + 2j, max_iter=200):
    # Crear la malla de valores complejos    
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    # Convertir a números complejos        
    # Inicializar la matriz de iteraciones
    iteration_counts = np.zeros(Z.shape, dtype=int)
    
    # Iterar la función hasta alcanzar el límite o la divergencia    
    for i in range(max_iter): 
        Z = np.exp(Z) + c 
        # Se aplica la función exponencial y se suma la constante c
        mask = np.abs(Z) < 1e10
        iteration_counts += mask # Contar iteraciones con puntos no divergentes

    return iteration_counts

def plot_map_exp():
    # Generar el fractal
    fractal = exponential_map_fractal()
    
    # Graficar el fractal
    plt.imshow(fractal, extent=(-2, 2, -2, 2), cmap='hot', interpolation='bilinear')
    plt.colorbar(label='Número de iteraciones')
    plt.title('Mapa exponencial')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.show()

plot_map_exp()