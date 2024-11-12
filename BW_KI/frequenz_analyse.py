import librosa
import numpy as np
import os

# Funktion, um die FFT durchzuführen und die Amplituden bestimmter Frequenzen zu extrahieren
def extract_frequencies(audio_files, target_frequencies, duration=0.5, output_file='all_frequencies.txt'):
    # Initialisieren eines Dictionaries, um die Amplituden für jede Frequenz über alle Dateien hinweg zu speichern
    frequency_amplitudes = {freq: [] for freq in target_frequencies}

    for audio_file in audio_files:
        # Laden der MP3-Datei
        y, sr = librosa.load(audio_file, sr=None)  # sr=None, um die Original-Sampling-Rate beizubehalten
        total_duration = librosa.get_duration(y=y, sr=sr)
        
        # Berechnen der Zeitstempel, bei denen wir die FFT berechnen wollen
        time_steps = np.arange(0, total_duration, duration)

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

    # Speichern der Amplituden in einer einzelnen Textdatei
    with open(output_file, 'w') as f:
        for freq in target_frequencies:
            line = " ".join([f"{amplitude:.6f}" for amplitude in frequency_amplitudes[freq]])
            f.write(line + '\n')
    
    print(f"Die Amplituden der Frequenzen wurden in '{output_file}' gespeichert.")

# Beispielaufruf
# Angenommen, die Audiodateien heißen '0.mp3', '1.mp3', '2.mp3', '3.mp3' im selben Verzeichnis
audio_files = [f"{i}.mp3" for i in range(4)]  # Liste der MP3-Dateien: '0.mp3', '1.mp3', '2.mp3', '3.mp3'
target_frequencies = [440, 880, 1320]  # Beispiel: Frequenzen in Hz (z.B. A4, A5, A6)
output_file = 'all_frequencies.txt'  # Ausgabe-Datei für alle Frequenzen

# Die Funktion aufrufen
extract_frequencies(audio_files, target_frequencies, duration=0.05, output_file=output_file)
