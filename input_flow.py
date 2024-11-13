import numpy as np
import pyaudio
import struct
from scipy.signal import stft
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioStream(object):
    def __init__(self):
        # Stream-Konstanten
        self.CHUNK = 1024 * 2
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 48000
        self.pause = False
        
        # Abtastrate in Sekunden pro Abfrage
        self.DELAYS = 0.05
        # Ziel-Frequenzen in Hz
        self.FREQUENCIES = np.array([
            50.,
            100.,
            200.,
            300.,
            440.,
            500.,
            700.,
            1000.,
            1500.,
            2000.,
            3000.,
            5000.,
            6000.
        ])
        
        # Lautstärken für jede Frequenz über die Zeit speichern
        self.values = [np.zeros(13) for _ in range(13)]
        
        # PyAudio initialisieren
        self.p = pyaudio.PyAudio()
        # Audio-Stream öffnen
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            output=True,
            frames_per_buffer=self.CHUNK,
        )
        self.start()


    def start(self):
        print('Stream gestartet')
        frame_count = 0
        start_time = time.time()
        model = load_model()

        while not self.pause:
            s_time = time.perf_counter()

            # Audio-Daten lesen
            data = np.frombuffer(self.stream.read(self.CHUNK), dtype=np.float32)
            #data_int = struct.unpack(str(2 * self.CHUNK) + 'B', data)
            data_array = np.array(data, dtype=np.float32) - 128  # In Signed Integer umwandeln

            # STFT anwenden
            freqs, times, Zxx = stft(data_array, fs=self.RATE, nperseg=self.CHUNK)
            Zxx = np.abs(Zxx)  # Amplitude

            # Lautstärken in dB für die Ziel-Frequenzen berechnen
            freq_amplitudes = []
            for target_freq in self.FREQUENCIES:
                idx = (np.abs(freqs - target_freq)).argmin()  # Finde den Index der Ziel-Frequenz
                amplitude = Zxx[idx, :]  # Amplitude über die Zeit
                decibel = 20 * np.log10(np.mean(amplitude) + 1e-6)  # Mittelwert + dB-Umrechnung
                freq_amplitudes.append(decibel)

            # Speichere die dB-Werte über die Zeit
            self.values.pop(0)
            self.values.append(freq_amplitudes)
            
            # Vorhersage mit dem MLP-Modell
            prediction = predict(torch.from_numpy(np.array(self.values).transpose().astype(np.float32)), model)

            if 0 in prediction:
                print(prediction)
            else:
                print("0")
            
            frame_count += 1

            if time.time() - start_time >= 100:
                self.pause = True

            if time.perf_counter() - s_time <= self.DELAYS:
                time.sleep(self.DELAYS - (time.perf_counter() - s_time))

        # Durchschnittliche Frequenzrate
        self.fr = frame_count / (time.time() - start_time)
        print('Durchschnittliche Operationsrate = {:.0f} pro Sekunde'.format(self.fr))
        self.exit_app()

    def exit_app(self):
        print('Stream geschlossen')
        self.p.terminate()  # Beende PyAudio korrekt

def predict(input, model):
    with torch.no_grad():
        pred = model(input)
    x = F.softsign(pred)  # Anwenden von softsign
    return (x > 0.5).int().numpy()

def load_model() -> object:
    output_size = 1
    input_size = 13
    
    class MLP(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)  # Letzte Schicht für binäre Klassifikation
            
        )

        def forward(self, x):
            return self.layers(x)

    model = MLP(input_size)
    model.load_state_dict(torch.load('model_cool.pt', weights_only=True))
    return model

if __name__ == '__main__':
    AudioStream()