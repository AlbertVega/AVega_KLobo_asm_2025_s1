import numpy as np
import matplotlib.pyplot as plt

def generate_sine_signal():
    t = np.linspace(0, 1, 500)  
    signal = np.sin(10 * np.pi * 400 * t)  
    return signal

def exponential_map_fractal(signal, xmin=-2, xmax=2, ymin=-2, ymax=2, width=800, height=800, max_iter=200):
    # Manejo de la señal
    fft = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(len(signal), 1/500)
    magnitude = np.abs(fft)
    
    frequency_component = frequencies[np.argmax(magnitude)] # Obtiene la frecuencia
    
    amplitude = np.max(signal) - np.min(signal) # Obtiene la amplitud
    
    phase = np.angle(fft[np.argmax(magnitude)]) # Obtiene la fase
    
    # Definir la constante c
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
    fractal, c = exponential_map_fractal(signal)
    
    # Graficar el fractal
    plt.figure(figsize=(10, 8))
    plt.imshow(fractal, extent=(-2, 2, -2, 2), cmap='turbo', interpolation='bilinear')
    plt.colorbar(label='Número de iteraciones')
    plt.title(f'Fractal Exponencial zn+1 = exp(zn) + c, donde c = {c:.4f}')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.show()

if __name__ == "__main__":
    signal = generate_sine_signal() # Funcion seno
    plot_map_exp(signal) # Grafica los puntos