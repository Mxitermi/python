import pyaudio
import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

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
    frames_per_buffer=CHUNK
)

highlits = np.array(np.arange(100, 3000, 15))

# Plot initialisieren
fig, ((ax, ax2),( ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 7))


# X-Achse für Zeit- und Frequenzbereich
x = np.arange(0, CHUNK)
x_fft = np.linspace(0, RATE, CHUNK)
x_point = np.linspace(0, 2000, len(highlits))

# Linien für den Waveform und das Frequenzspektrum
line, = ax.plot(x, np.random.rand(CHUNK), "-", lw=2)
line_fft, = ax2.semilogx(x_fft, np.random.rand(CHUNK), "-", lw=2)
points_indiv = ax3.scatter(x_point, np.random.rand(len(highlits)), s=4, c=np.random.rand(len(highlits)))

# Grenzen für den Waveform-Plot (Amplitude)
scaler = 3200

ax.set_ylim(-scaler, scaler)
ax.set_xlim(0, CHUNK)
ax.set_title("Waveform")
ax.set_xlabel("Samples")
ax.set_ylabel("Amplitude")

# Grenzen für das Frequenzspektrum
ax2.set_xlim(20, RATE / 2)
ax2.set_title("Frequenzspektrum (FFT)")
ax2.set_xlabel("Frequenz (Hz)")
ax2.set_ylabel("Amplitude")

ax3.set_ylim(-0.2, 1)
ax3.set_xlim(0, 2100)
ax3.set_xlabel("Frequenz (Hz)")


# Interaktive Plots
plt.show(block=False)
fig.show()
while True:
    if not plt.fignum_exists(fig.number):
        print("Programm beendet.")
        break
    
    data = stream.read(CHUNK)

    
    
    ax2.plot([highlits,highlits], [0,1], color ='green', linewidth=1.5, linestyle="--")
    # Vertikale gestrichelte Linie an der Stelle 'pos' einzeichnen
    # (von der x-Achse bis zum Graph der cos-Funktion):
    

    # Umwandeln in 16-Bit signed Integer
    data_int = np.array(struct.unpack(str(CHUNK) + 'h', data), dtype='h')
    
    # Überlauf verhindern und Wertebereich auf -500 bis 499 beschränken
    data_int = (data_int + scaler - 1) % (scaler * 2) - scaler
    line.set_ydata(data_int)
    
    
    # FFT berechnen und Amplituden holen
    
    y_fft = fft(data_int)
    y_fft = np.abs(y_fft[0:CHUNK]) * 2 / (CHUNK * 1000)  #Normalisieren
    
    # Update der FFT-Daten
    data_int_y = y_fft[highlits]
    points_indiv.set_array( data_int_y)
    points_indiv.set_offsets(np.c_[highlits, data_int_y])
    line_fft.set_ydata(y_fft)
    fig.canvas.draw()
    fig.canvas.flush_events()
