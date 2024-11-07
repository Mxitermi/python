import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from scipy import signal

# PyAudio-Konfiguration
FORMAT = pyaudio.paInt16  # 16-Bit Auflösung
CHANNELS = 1              # Mono
RATE = 44100               # Abtastrate (Hz)
CHUNK = 1024 * 4              # Frames pro Buffer
DEVICE_INDEX_INPUT = 2     # Mikrofon Eingabegerät
DEVICE_INDEX_OUTPUT = 4    # Lautsprecher Ausgabegerät

# Notch-Filter-Konfiguration
notch_freq = 440  # Frequenz, die entfernt werden soll (Hz)
quality_factor = 50.0  # Qualität des Filters
b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, RATE)

# Audio-Stream öffnen
p = pyaudio.PyAudio()
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    input_device_index=DEVICE_INDEX_INPUT,
    output_device_index=DEVICE_INDEX_OUTPUT,
    frames_per_buffer=CHUNK
)


print("Starte Mikrofon-Streaming...")
try:
    while True:
        # Audio Daten vom Mikrofon aufnehmen
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)  # Konvertierung in numpy-Array


        
        # Notch-Filter auf das Mikrofon-Signal anwenden
        y_notched = signal.filtfilt(b_notch, a_notch, audio_data)




        # Wiedergabe des gefilterten Signals (optional)
        filtered_output = y_notched.astype(np.int16).tobytes()
        stream.write(filtered_output)

except KeyboardInterrupt:
    print("Streaming beendet")

finally:
    # Beende den Audio-Stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    plt.ioff()  # Deaktiviert interaktiven Modus
    plt.show()
