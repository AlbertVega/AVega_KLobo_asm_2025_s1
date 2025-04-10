import numpy as np
import matplotlib.pyplot as plt

def newton_map(signal, width=800, height=800, max_iter=30, tol=1e-6):
    signal = np.asarray(signal)
    
    # verificar que la señal tenga al menos algunos valores
    if len(signal) < 2:
        raise ValueError("La señal debe tener al menos dos elementos.")

    # Normaliza la señal para que esté entre -2 y 2
    signal = 4 * (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) - 2

    # Crea una malla compleja a partir de la señal
    x = np.linspace(-2, 2, width)
    y = np.interp(np.linspace(0, len(signal)-1, height), np.arange(len(signal)), signal)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    roots = np.array([1, np.exp(2j * np.pi / 3), np.exp(-2j * np.pi / 3)])
    colors = np.zeros(Z.shape, dtype=int)

    for _ in range(max_iter):
        Z -= (Z**3 - 1) / (3 * Z**2)

    for j, root in enumerate(roots):
        colors[np.abs(Z - root) < tol] = j + 1

    return colors

def plot_newton_map(signal):
    fractal = newton_map(signal)
    plt.imshow(fractal, cmap='viridis', extent=(-2, 2, -2, 2))
    plt.colorbar()
    plt.title("Fractal de Newton modulado por señal")
    plt.show()