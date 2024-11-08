import pyaudio
import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import signal  # Für den Notch-Filter

# Parameter
CHUNK = 1024 * 4
FORMAT = pyaudio.paInt16  # 16-Bit signed Integer
CHANNELS = 1
RATE = 44100

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

# Notch-Filter für die Ziel-Frequenz erstellen
# Beispiel: Unterdrücken einer Frequenz von 400 Hz
target_freq = 400  # Frequenz, die wir unterdrücken wollen
quality_factor = 50  # Bestimmt die Bandbreite des Filters (je höher, desto schmaler)
b, a = signal.iirnotch(target_freq, quality_factor, RATE)

print("Listening...")


while True:
    

    # Lese Daten vom Mikrofon
    data = stream.read(CHUNK)

    # Umwandeln in 16-Bit signed Integer
    data_int = np.array(struct.unpack(str(CHUNK) + 'h', data), dtype='h')

    # Notch-Filter anwenden
    data_filtered = signal.filtfilt(b, a, data_int)

    # Sicherstellen, dass die gefilterten Daten das richtige Format haben (int16)
    data_filtered = np.asarray(data_filtered, dtype=np.int16)

    # FFT für das gefilterte Signal berechnen
    #y_fft = fft(data_filtered)

    # Ausgabedaten an das Audio-Interface
    data_out_ = data_filtered.tobytes()  # In Bytes umwandeln

    # Daten an das Ausgabegerät senden
    stream.write(data)
