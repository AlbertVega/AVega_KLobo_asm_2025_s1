import numpy as np
import matplotlib.pyplot as plt

def exponential_map(signal, xmin=-2, xmax=2, ymin=-2, ymax=2, width=800, height=800, max_iter=200):
    # 1. Calcular la FFT para obtener la frecuencia dominante
    fft = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(len(signal), 1/500)
    magnitude = np.abs(fft)
    
    # Frecuencia dominante
    frequency_component = frequencies[np.argmax(magnitude)]
    
    # 2. Calcular la amplitud
    amplitude = np.max(signal) - np.min(signal)
    
    # 3. Obtener la fase del componente dominante
    phase = np.angle(fft[np.argmax(magnitude)])
    
    # Usar la fórmula especificada para definir la constante c
    c = frequency_component / 100 + amplitude * np.exp(1j * phase)
    
    # Crear la malla de valores complejos
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # Inicializar la matriz de iteraciones
    iteration_counts = np.zeros(Z.shape, dtype=int)
    
    # Iterar la función hasta alcanzar el límite o la divergencia
    for i in range(max_iter): 
        with np.errstate(over='ignore', invalid='ignore'):
            Z = np.exp(Z) + c 
            # Se aplica la función exponencial y se suma la constante c
            mask = np.abs(Z) < 1e10
            iteration_counts += mask # Contar iteraciones con puntos no divergentes

    return iteration_counts, c

def plot_map_exp(signal):
    # Generar el fractal usando la señal
    fractal, c = exponential_map(signal)
    
    # Graficar el fractal
    plt.figure(figsize=(10, 8))
    plt.imshow(fractal, extent=(-2, 2, -2, 2), cmap='turbo', interpolation='bilinear')
    plt.colorbar(label='Número de iteraciones')
    plt.title(f'Fractal Exponencial con c = {c:.4f}')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.show()
    