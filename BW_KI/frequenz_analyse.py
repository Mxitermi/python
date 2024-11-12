import librosa
import numpy as np
import scipy.fft

# Funktion, um die FFT durchzuführen und die Amplituden bestimmter Frequenzen zu extrahieren
def extract_frequencies(audio_file, target_frequencies, duration=0.5, output_file='frequencies.txt'):
    # Laden der MP3-Datei
    y, sr = librosa.load(audio_file, sr=None)  # sr=None, um die Original-Sampling-Rate beizubehalten
    total_duration = librosa.get_duration(y=y, sr=sr)
    
    # Berechnen der Zeitstempel, bei denen wir die FFT berechnen wollen
    time_steps = np.arange(0, total_duration, duration)

    # Initialisieren eines Dictionaries, um die Amplituden für jede Frequenz zu speichern
    frequency_amplitudes = {freq: [] for freq in target_frequencies}

    for time in time_steps:
        # Bestimmen des Start- und Endindex für das Zeitfenster
        start_sample = librosa.time_to_samples(time, sr=sr)
        end_sample = librosa.time_to_samples(time + duration, sr=sr)
        
        # Auslesen des Audiosignals im aktuellen Zeitfenster
        audio_segment = y[start_sample:end_sample]
        
        # Durchführung der FFT (Fast Fourier Transformation)
        fft_result = np.fft.fft(audio_segment)
        fft_freqs = np.fft.fftfreq(len(fft_result), 1/sr)
        
        # Berechnung der Amplituden (Betrag der FFT)
        amplitudes = np.abs(fft_result)
        
        # Extraktion der Amplituden für die gewünschten Frequenzen
        for freq in target_frequencies:
            # Finden der Indexposition der nächsten Frequenz
            closest_idx = np.argmin(np.abs(fft_freqs - freq))
            frequency_amplitudes[freq].append(amplitudes[closest_idx])
    
    # Speichern der Amplituden in der Textdatei
    with open(output_file, 'w') as f:
        for freq in target_frequencies:
            line = " ".join([f"{amplitude:.6f}" for amplitude in frequency_amplitudes[freq]]) 
            f.write(line + '\n')

# Beispielaufruf
audio_file = 'audio_input_f.wav'  # Pfad zur MP3-Datei
target_frequencies = [210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 320, 340, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700]  # Beispiel: Frequenzen in Hz (z.B. A4, A5, A6)
output_file = 'frequencies.txt'  # Ausgabedatei
#directory = "C:/daten/python/Samples0_real/"
#filepaths = [os.path.join(directory, f"{i}.mp3") for i in range(0, 1)]

# Die Funktion aufrufen
extract_frequencies(audio_file, target_frequencies, duration=0.05, output_file=output_file)

print(f"Die Amplituden der Frequenzen wurden in '{output_file}' gespeichert.")
