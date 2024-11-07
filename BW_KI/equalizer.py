import pyaudio
import numpy as np
from scipy.fftpack import fft, fftfreq, ifft

# Parameter
CHUNK = 1024 * 4
FORMAT = pyaudio.paInt16  # 16-Bit signed Integer
CHANNELS = 1
RATE = 44100
FREQUENCY = 500  # Ziel-Frequenz in Hz (die zu stummschaltende Frequenz)

# PyAudio initialisieren
p = pyaudio.PyAudio()

# Stream öffnen
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    input_device_index=2,
    output_device_index=4,
    frames_per_buffer=CHUNK
)

# Phase für den Sinuston (wird über mehrere Schleifen aufrechterhalten)
phase = 0
omega = 2 * np.pi * FREQUENCY / RATE  # Kreisfrequenz für den Sinuston

print("Listening...")

try:
    while True:
        # Lese Daten vom Mikrofon
        data = stream.read(CHUNK)
        audio_data = np.frombuffer(data, dtype=np.int16)

        # FFT auf die Daten anwenden
        N = len(audio_data)
        yf = fft(audio_data)
        xf = fftfreq(N, 1 / RATE)[:N // 2]  # Nur positive Frequenzen behalten

        # Index der Ziel-Frequenz finden
        index = np.argmin(np.abs(xf - FREQUENCY))

        # Amplitude der Ziel-Frequenz berechnen
        print(index)
        yf[:N // 2][index] = 0
        for i in range(0, 20):
            yf[:N // 2][index - 20 + i] = 0

        recovered_signal = ifft(yf)

        # Da die iFFT theoretisch komplexe Zahlen ausgibt (durch numerische Ungenauigkeiten),
        # musst du den reellen Teil des Ergebnisses nehmen:
        recovered_signal = np.real(recovered_signal).astype(np.int16)

        # Ausgabedaten an das Audio-Interface senden
        data_out_ = recovered_signal.tobytes()
        stream.write(data_out_)

except KeyboardInterrupt:
    print("Streaming beendet.")

finally:
    # Stream schließen
    stream.stop_stream()
    stream.close()
    p.terminate()
