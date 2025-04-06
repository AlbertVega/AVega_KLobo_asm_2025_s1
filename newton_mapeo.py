import numpy as np
import matplotlib.pyplot as plt

def newton_fractal(width=800, height=800, xmin=-2, xmax=2, ymin=-2, ymax=2, max_iter=30, tol=1e-6):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    roots = np.array([1, np.exp(2j * np.pi / 3), np.exp(-2j * np.pi / 3)])
    colors = np.zeros(Z.shape, dtype=int)
    
    for i in range(max_iter):
        Z -= (Z**3 - 1) / (3 * Z**2)  # Método de Newton-Raphson
        
    for j, root in enumerate(roots):
        colors[np.abs(Z - root) < tol] = j + 1
    
    return colors

def plot_fractal():
    fractal = newton_fractal()
    plt.imshow(fractal, cmap='viridis', extent=(-2, 2, -2, 2))
    plt.colorbar()
    plt.title("Fractal de Newton para f(z) = z^3 - 1")
    plt.show()

plot_fractal()
