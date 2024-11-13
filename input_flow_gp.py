import torch
import librosa
import numpy as np
import pyaudio
import torch.nn as nn

# Parameter für das Audio-Streaming
SAMPLE_RATE = 44100  # Sampling-Rate für Mikrofon-Stream
CHUNK_SIZE = 4096  # Anzahl der Samples pro Chunk für Echtzeitverarbeitung
DURATION = CHUNK_SIZE / SAMPLE_RATE  # Dauer jedes Chunks in Sekunden

# Modellparameter
input_size = 13  # Anzahl der Frequenzmerkmale pro Datensatz (13 Merkmale)

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
model.load_state_dict(torch.load('model_cool.pt', weights_only=True))
model.eval()

# Liste der Frequenzen, die das Modell analysieren soll (13 Werte als Beispiel)
target_frequencies = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300]  # Beispiel-Frequenzen

# Funktion zur FFT und Extraktion der Amplituden der gewünschten Frequenzen
def extract_amplitudes(audio_segment, sr, target_frequencies):
    # FFT durchführen
    fft_result = np.fft.fft(audio_segment)
    fft_freqs = np.fft.fftfreq(len(fft_result), 1 / sr)
    amplitudes = np.abs(fft_result)

    # Extraktion der Amplituden für die gewünschten Frequenzen
    frequency_amplitudes = []
    for freq in target_frequencies:
        # Finden der Indexposition der nächsten Frequenz
        closest_idx = np.argmin(np.abs(fft_freqs - freq))
        frequency_amplitudes.append(amplitudes[closest_idx])
    return np.array(frequency_amplitudes, dtype=np.float32)

# Initialisiere PyAudio für den Audio-Stream
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                output=True,
                input_device_index=2,
                output_device_index=4,
                frames_per_buffer=CHUNK_SIZE)

try:
    print("Monitoring for feedback... Press Ctrl+C to stop.")
    while True:
        # Audio-Daten vom Mikrofon lesen
        audio_data = np.frombuffer(stream.read(CHUNK_SIZE), dtype=np.float32)

        # Amplituden der Ziel-Frequenzen extrahieren
        amplitudes = extract_amplitudes(audio_data, SAMPLE_RATE, target_frequencies)

        # Überprüfen, ob genügend Merkmale vorhanden sind, andernfalls überspringen
        if len(amplitudes) != input_size:
            continue

        # Amplituden normalisieren und in Tensor umwandeln
        input_tensor = torch.from_numpy(amplitudes).unsqueeze(0)  # Batch-Dimension hinzufügen

        # MLP-Modell auf die extrahierten Frequenzdaten anwenden
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.sigmoid(output).item()

        # Falls das Modell Rückkopplung erkennt, Frequenzen ausgeben
        if prediction > 0.5:  # Beispiel-Schwellenwert für Rückkopplung
            detected_freqs = [f for f, amp in zip(target_frequencies, amplitudes) if amp > 0.1]
            print(f"Feedback detected at frequencies: {detected_freqs} Hz")

except KeyboardInterrupt:
    print("Stopping...")

finally:
    # Audio-Stream und PyAudio beenden
    stream.stop_stream()
    stream.close()
    p.terminate()
