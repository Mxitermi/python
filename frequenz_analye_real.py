import numpy as np
import librosa
import os

class AudioAnalyzer:
    def __init__(self, filepath, target_frequencies):
        self.filepath = filepath
        self.target_frequencies = np.array(target_frequencies)
        self.audio_data = None
        self.sample_rate = None
        self.amplitude_levels_over_time = {}
        self.time_stamps = None

    def load_audio(self):
        # Lade die Audiodatei mit librosa
        self.audio_data, self.sample_rate = librosa.load(self.filepath, sr=None)
        print(f"Audio geladen: {self.filepath}, Sample-Rate: {self.sample_rate} Hz")

    def analyze_frequencies(self):
        # Berechne die STFT (Short-Time Fourier Transform)
        stft_result = np.abs(librosa.stft(self.audio_data))
        # Frequenz-Bins und Zeitpunkte
        freqs = librosa.fft_frequencies(sr=self.sample_rate)
        times = librosa.frames_to_time(np.arange(stft_result.shape[1]), sr=self.sample_rate)
        
        # Finde die Frequenzen in den Frequenz-Bins
        for target_freq in self.target_frequencies:
            idx = (np.abs(freqs - target_freq)).argmin()
            # Extrahiere die Amplitude für die Ziel-Frequenz über die Zeit
            amplitude_over_time = stft_result[idx, :]
            self.amplitude_levels_over_time[target_freq] = amplitude_over_time
        self.time_stamps = times

    def get_results(self):
        results = []
        for freq in self.target_frequencies:
            decibels = self.amplitude_levels_over_time.get(freq, [])
            results.append(' '.join([f'{db:.1f}' for db in decibels]))
        return results

def process_files(filepaths, target_frequencies, output_filename):
    with open(output_filename, 'w') as f:
        for filepath in filepaths:
            # Erstelle das AudioAnalyzer-Objekt für jede Datei
            analyzer = AudioAnalyzer(filepath, target_frequencies)
            analyzer.load_audio()
            analyzer.analyze_frequencies()
            
            # Ergebnisse für die aktuelle Datei
            results = analyzer.get_results()
            for result in results:
                f.write(result + '\n')

    print(f"Ergebnisse in Datei '{output_filename}' gespeichert.")

# Definiere die Frequenzen, die du analysieren möchtest
target_frequencies = [247,262,277,294,311,330,349,370,380,392,415,420,430,440,450,460,466,470,480,490,494,500,510,523,554,587,622]

# Verzeichnis mit den Audiodateien
directory = "C:/daten/python/Samples0_real/"
filepaths = [os.path.join(directory, f"{i}.mp3") for i in range(0, 1)]

# Datei, in der die Ergebnisse gespeichert werden
output_filename = "frequenz_analyse_real.txt"

# Verarbeite die Dateien und speichere die Ergebnisse
process_files(filepaths, target_frequencies, output_filename)
