import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt

def generate_spectrogram(signal, sample_rate, nperseg=1024):
    """
    Genera y grafica el espectrograma complejo (magnitud y fase) de una señal usando STFT.
    
    Parámetros:
        signal (array): Señal de entrada en el dominio del tiempo.
        sample_rate (int): Frecuencia de muestreo (Hz).
        nperseg (int): Tamaño de la ventana para la STFT (por defecto: 256).
    """
    # Calcula la STFT (retorna frecuencias, tiempos y Zxx = espectrograma complejo)
    f, t, Zxx = stft(signal, fs=sample_rate, nperseg=nperseg, window='hann', return_onesided=False)
    
    # Extrae magnitud y fase del espectrograma complejo
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)  # Fase en radianes
    
    # Grafica la magnitud (espectrograma clásico en escala logarítmica)
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.pcolormesh(t, np.fft.fftshift(f), 20 * np.log10(np.fft.fftshift(magnitude, axes=0)), shading='gouraud')
    plt.colorbar(label='Magnitud (dB)')
    plt.title('Espectrograma (Magnitud)')
    plt.ylabel('Frecuencia (Hz)')
    plt.xlabel('Tiempo (s)')
    
    # Grafica la fase
    plt.subplot(2, 1, 2)
    plt.pcolormesh(t, np.fft.fftshift(f), np.fft.fftshift(phase, axes=0), shading='gouraud')
    plt.colorbar(label='Fase (rad)')
    plt.title('Espectrograma (Fase)')
    plt.ylabel('Frecuencia (Hz)')
    plt.xlabel('Tiempo (s)')
    
    plt.tight_layout()
    plt.show()

def generate_tone(frequency, duration, sample_rate=40960):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * frequency * t), sample_rate

def generate_composite_tone(frequencies, duration, sample_rate=40960):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = sum(np.sin(2 * np.pi * f * t) for f in frequencies)
    return signal / len(frequencies), sample_rate

# Ejemplo de uso
if __name__ == "__main__":
    # Generar un tono simple de 1 kHz (A4)
    tone, sr = generate_tone(1000, 2)
    generate_spectrogram(tone, sr)
    
    # Generar un tono compuesto (sumando 440 Hz, 880 Hz, 1000 Hz y 10 Hz)
    composite_tone, sr = generate_composite_tone([440, 880, 1000, 10], 2)
    generate_spectrogram(composite_tone, sr)
