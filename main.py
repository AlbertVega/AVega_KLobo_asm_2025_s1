import espectrograma
import exp_mapeo
import newton_mapeo
import numpy as np

# función que genera una señal simple de seno, recibe la frecuencia
def generate_sine_signal(frequency=5, duration=1, sample_rate=800):
    t = np.linspace(0, duration, sample_rate) 
    signal = np.sin(2 * np.pi * frequency * t)  
    return signal

if __name__ == "__main__":
    # ------------------------- ESPECTROGRAMA -------------------------------------
    # Desde archivo WAV
    wav_signal, wav_sr = espectrograma.read_wav_file("VOLTAGE.wav") 
    wav_signal2, wav_sr2 = espectrograma.read_wav_file("DLLT.wav") 
    espectrograma.generate_spectrogram(wav_signal, wav_sr)
    espectrograma.generate_spectrogram(wav_signal2, wav_sr2)

    # Generar un tono simple de 1 kHz (A4)
    tone, sr = espectrograma.generate_tone(1000, 2)
    espectrograma.generate_spectrogram(tone, sr)
    
    # Generar un tono compuesto (sumando 440 Hz, 880 Hz, 1000 Hz y 10 Hz)
    composite_tone, sr = espectrograma.generate_composite_tone([440, 880, 1000, 10], 2)
    espectrograma.generate_spectrogram(composite_tone, sr)

    # -------------- GENERAR SENO QUE SE UTILZA PARA MAPEO -----------------------
    signal_newton = generate_sine_signal(800)  # frecuencia de 800 Hz
    signal_exp = generate_sine_signal(2000)    # frecuencia de 2k Hz

    # ------------------------- MAPEO NEWTON -------------------------------------
    #newton_mapeo.plot_newton_map(signal_newton) # aplicar mapeo de newton a la señal
    newton_mapeo.plot_newton_map(wav_signal) # aplicar mapeo de newton a la señal
    newton_mapeo.plot_newton_map(wav_signal2) # aplicar mapeo de newton a la señal

    # ------------------------- MAPEO EXPONENCIAL --------------------------------
    #exp_mapeo.plot_map_exp(signal_exp) # aplicar mapeo exponencial a la señal
    exp_mapeo.plot_map_exp(wav_signal)
    exp_mapeo.plot_map_exp(wav_signal2)
    
