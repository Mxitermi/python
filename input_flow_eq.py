import torch
import torch.nn as nn
import librosa
import numpy as np
import pyaudio
from collections import deque
from scipy.signal import iirnotch, lfilter

# Parameter für das Audio-Streaming
SAMPLE_RATE = 44100
CHUNK_SIZE = 4096
STACK_SIZE = 13  # Länge des Rolling Windows für jede Frequenz

# Modellparameter
input_size = STACK_SIZE  # Das Modell wird jetzt nur den Verlauf einer Frequenz betrachten

# Definieren der Modellarchitektur (entsprechend dem Trainingsskript)
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Ausgabe für binäre Klassifikation
        )

    def forward(self, x):
        return self.layers(x)

# Lade das trainierte Modell und setze es in den Evaluierungsmodus
model = MLP(input_size)
model.load_state_dict(torch.load("model_cool.pt"))
model.eval()

# Ziel-Frequenzen und Rolling Window für jede Frequenz
target_frequencies = [121, 145, 186, 225, 259, 281, 347, 402, 415, 440, 479, 520, 543, 580, 629, 676,  741, 792, 853, 901, 982, 1035, 1189]
frequency_stacks = {freq: deque(maxlen=STACK_SIZE) for freq in target_frequencies}  # Erstellen eines Rolling Stacks

# Funktion zur FFT und Extraktion der Amplituden für jede gewünschte Frequenz
def extract_amplitudes(audio_segment, sr, target_frequencies):
    fft_result = np.fft.fft(audio_segment)
    fft_freqs = np.fft.fftfreq(len(fft_result), 1 / sr)
    amplitudes = np.abs(fft_result)
    frequency_amplitudes = {}

    # Amplituden für Ziel-Frequenzen extrahieren
    for freq in target_frequencies:
        closest_idx = np.argmin(np.abs(fft_freqs - freq))
        frequency_amplitudes[freq] = amplitudes[closest_idx]
    return frequency_amplitudes

# Funktion für einen Band-Stop-Filter (Notch Filter)
def apply_notch_filter(audio_data, sample_rate, target_freq, quality_factor=30):
    b, a = iirnotch(target_freq, quality_factor, sample_rate)
    filtered_data = lfilter(b, a, audio_data)
    return filtered_data

# Initialisiere PyAudio für den Audio-Stream
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                output=True,  # Damit der gefilterte Ton ausgegeben werden kann
                frames_per_buffer=CHUNK_SIZE)

try:
    print("Monitoring for feedback... Press Ctrl+C to stop.")
    while True:
        # Audio-Daten vom Mikrofon lesen
        audio_data = np.frombuffer(stream.read(CHUNK_SIZE), dtype=np.float32)

        # Amplituden der Ziel-Frequenzen extrahieren
        amplitudes = extract_amplitudes(audio_data, SAMPLE_RATE, target_frequencies)

        # Stack aktualisieren und prüfen
        for freq, amplitude in amplitudes.items():
            # Neuen Amplitudenwert zum Stack hinzufügen
            frequency_stacks[freq].append(amplitude)
            
            # Prüfen, ob der Stack für diese Frequenz die richtige Länge hat
            if len(frequency_stacks[freq]) == STACK_SIZE:
                # Eingabe für das Modell vorbereiten
                input_tensor = torch.tensor(list(frequency_stacks[freq]), dtype=torch.float32).unsqueeze(0)
                
                # Modellvorhersage durchführen
                with torch.no_grad():
                    output = model(input_tensor)
                    prediction = torch.sigmoid(output).item()
                
                # Falls Rückkopplung erkannt wird, Frequenz stummschalten
                if prediction > 0.8:  # Beispiel-Schwellenwert
                    print(f"Rückkopplung erkannt bei {freq} Hz. Filter wird angewendet.")
                    audio_data = apply_notch_filter(audio_data, SAMPLE_RATE, freq)

        # Das gefilterte Audiosignal ausgeben
        stream.write(audio_data.astype(np.float32).tobytes())

except KeyboardInterrupt:
    print("Stopping...")

finally:
    # Audio-Stream und PyAudio beenden
    stream.stop_stream()
    stream.close()
    p.terminate()