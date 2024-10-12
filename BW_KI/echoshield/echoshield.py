import numpy as np
import librosa
import os
import torch
import torch.nn as nn

class AudioAnalyzer:
    def __init__(self, filepath, target_frequencies):
        self.filepath = filepath
        self.target_frequencies = np.array(target_frequencies)
        self.audio_data = None
        self.sample_rate = None
        self.decibel_levels_over_time = {}
        self.time_stamps = None

    def load_audio(self):
        self.audio_data, self.sample_rate = librosa.load(self.filepath, sr=None)
        print(f"Audio geladen: {self.filepath}, Sample-Rate: {self.sample_rate} Hz")

    def analyze_frequencies(self):
        stft_result = np.abs(librosa.stft(self.audio_data))
        freqs = librosa.fft_frequencies(sr=self.sample_rate)
        times = librosa.frames_to_time(np.arange(stft_result.shape[1]), sr=self.sample_rate)

        for target_freq in self.target_frequencies:
            idx = (np.abs(freqs - target_freq)).argmin()
            amplitude_over_time = stft_result[idx, :]
            decibels_over_time = 20 * np.log10(amplitude_over_time + 1e-6)
            self.decibel_levels_over_time[target_freq] = decibels_over_time
        self.time_stamps = times

    def get_results(self):
        results = []
        for freq in self.target_frequencies:
            decibels = self.decibel_levels_over_time.get(freq, [])
            results.append(' '.join([f'{db:.1f}' for db in decibels]))
        return results

def process_file(filepath, target_frequencies, output_filename):
    analyzer = AudioAnalyzer(filepath, target_frequencies)
    analyzer.load_audio()
    analyzer.analyze_frequencies()
    results = analyzer.get_results()

    with open(output_filename, 'w') as f:
        for result in results:
            f.write(result + '\n')
    print(f"Ergebnisse in Datei '{output_filename}' gespeichert.")

def main():
    input_file = "frequenz_analyse.txt"
    output_file = "frequenz_analyse.txt"

    def kürze_auf_vielfaches_von_13(arr):
        länge = len(arr)
        kürzung = länge - (länge % 13)
        return arr[:kürzung]

    try:
        liste = []
        with open(input_file, 'r') as file:
            for line in file:
                r = line.split()
                gekürzte_liste = kürze_auf_vielfaches_von_13(r)
                new_line = ' '.join(gekürzte_liste)
                liste.append(new_line)

        with open(output_file, 'w') as file:
            for line in liste:
                file.write(line + '\n')
        print("Daten erfolgreich in die Datei geschrieben.")

    except FileNotFoundError as e:
        print(f"Datei nicht gefunden: {e}")
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")

def pred(input, model):
    with torch.no_grad():
        pred = model(input)
    pred = torch.sigmoid(pred)
    rounded_pred = torch.round(pred * 1000) / 1000
    return rounded_pred

def load_model():
    output_size = 1
    input_size = 13
    class MLP(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, output_size)
            )

        def forward(self, x):
            out = self.layers(x)
            return out

    model = MLP(input_size)
    model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu'), weights_only=True))
    return model

def process_predictions(filename, model):
    results = []
    with open(filename, 'r') as file:
        lines = file.readlines()

        for line in lines:
            values = list(map(float, line.strip().split()))
            label = values[-1]
            input_values = values[:-1]

            for i in range(0, len(input_values), 13):
                batch = input_values[i:i+13]
                if len(batch) < 13:
                    break
                input_tensor = torch.tensor([batch], dtype=torch.float32)
                prediction = pred(input_tensor, model)
                rounded_prediction = torch.round(prediction).item()
                results.append(rounded_prediction)

    return results

if __name__ == "__main__":
    target_frequencies = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000]
    directory = "."  # Aktueller Ordner

    # Auswahl der MP3-Datei
    print("Verfügbare MP3-Dateien:")
    filepaths = [f for f in os.listdir(directory) if f.endswith('.mp3')]
    
    if not filepaths:
        print("Keine MP3-Dateien im aktuellen Verzeichnis gefunden.")
    else:
        for idx, filename in enumerate(filepaths):
            print(f"{idx + 1}: {filename}")

        file_index = int(input("Bitte die Nummer der Datei wählen (1 bis {}): ".format(len(filepaths)))) - 1
        selected_file = os.path.join(directory, filepaths[file_index])
        
        output_filename = "frequenz_analyse.txt"
        
        process_file(selected_file, target_frequencies, output_filename)
        main()

        model = load_model()
        model.eval()
        filename = 'frequenz_analyse.txt'
        predictions = process_predictions(filename, model)
        frequencies = ['100Hz', "200Hz", "300Hz", "400Hz", "500Hz", "600Hz", "700Hz", "800Hz", "900Hz", "1000Hz", "2000Hz", "3000Hz", "4000Hz"]
        group_size = 13

        for freq_idx, freq in enumerate(frequencies):
            rueckkopplung_erkannt = False
            print(f"Frequenz: {freq}")

            start_idx = freq_idx * group_size
            end_idx = start_idx + group_size
            prediction_slice = predictions[start_idx:end_idx]
            for idx, wert in enumerate(prediction_slice):
                if wert > 0.5:
                    rueckkopplung_erkannt = True
                    zeitpunkt = ((idx) * 0.1) + 0.1
                    print(f"  Rückkopplung erkannt bei {zeitpunkt:.1f} Sekunden.")
            if not rueckkopplung_erkannt:
                print("  Keine Rückkopplung erkannt.")
    input("Drücke Enter, um das Programm zu schließen...")
