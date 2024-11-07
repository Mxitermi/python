import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import iirnotch, filtfilt

# Lade die WAV-Datei
sample_rate, data = wav.read('rinput_440hz.wav')

# Überprüfe, ob das Audio Stereo oder Mono ist
if len(data.shape) > 1:
    # Wähle den linken Kanal aus (falls Stereo)
    data = data[:, 0]

# Liste der Frequenzen, die gefiltert werden sollen
notch_freqs = [430, 440, 1320]  # Frequenzen in Hz, die du filtern möchtest
quality_factor = 30.0  # Qualität des Filters (je höher, desto schmaler der Filter)

# Wende Notch-Filter für jede Frequenz an
filtered_data = data.copy()  # Kopiere das Originalsignal, um es zu filtern

for notch_freq in notch_freqs:
    # Berechne die Notch-Filterkoeffizienten
    b, a = iirnotch(notch_freq, quality_factor, fs=sample_rate)
    
    # Wende den Filter auf das Audiosignal an
    filtered_data = filtfilt(b, a, filtered_data)

# Speichere die gefilterte WAV-Datei
output_file = 'output_filtered_multiple_freqs.wav'
wav.write(output_file, sample_rate, filtered_data.astype(np.int16))

print(f"Die gefilterte Audiodatei wurde unter '{output_file}' gespeichert.")
