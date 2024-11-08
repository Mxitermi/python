import pyaudio
import numpy as np
import sounddevice as sd
from scipy.fft import rfft, irfft, rfftfreq

# Einstellungen für die Audiobearbeitung
SAMPLE_RATE = 44100  # Abtastrate (44.1 kHz für hohe Qualität)
BLOCK_SIZE = 1024    # Blockgröße (kleinere Blöcke = geringere Latenz)

# Frequenz, die unterdrückt werden soll
TARGET_FREQUENCY = 440  # Ziel-Frequenz in Hz (A4)
MAX_GAIN = 0  # Verstärkung auf 0 setzen für die Ziel-Frequenz

# Identifiziere das virtuelle Gerät für die Audioeingabe (z.B., "CABLE Output")
input_device_name = sd.query_devices()[0]["name"]  # Virtuelles Audiogerät
output_device_name = sd.query_devices()[0]["name"]  # Standard-Lautsprecher

def get_device_index(device_name, kind='input'):
    """Finde die Geräte-ID eines spezifischen Geräts nach Namen."""
    if not device_name:  # Gerätename ist None oder leer
        return None

    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        if kind == 'input' and device_name in device['name'] and device['max_input_channels'] > 0:
            return idx
        elif kind == 'output' and device_name in device['name'] and device['max_output_channels'] > 0:
            return idx
    return None

# Finde die Geräte-IDs für die Eingabe und Ausgabe
input_device = get_device_index(input_device_name, kind='input')
output_device = get_device_index(output_device_name, kind='output')

# Überprüfen, ob die Geräte vorhanden sind
if input_device is None:
    raise ValueError(f"Eingabegerät '{input_device_name}' nicht gefunden.")
if output_device is None:
    print(f"Standardausgabegerät wird verwendet.")

def weighted_gain(frequency, current_freq, max_gain):
    """Berechnet die Verstärkung basierend auf der Distanz zur Ziel-Frequenz."""
    distance = abs(current_freq - frequency)
    if distance == 0:
        return 0  # Ziel-Frequenz unterdrücken
    elif distance <= 1:
        return max_gain * 0.2  # 20% Verstärkung
    elif distance <= 2:
        return max_gain * 0.5  # 50% Verstärkung
    elif distance <= 3:
        return max_gain * 0.9  # 90% Verstärkung
    else:
        return 1  # Keine Dämpfung

def apply_weighted_eq(data, sample_rate, target_frequency, max_gain):
    """Wendet den gewichteten Equalizer auf ein Datenblock an."""
    # FFT auf den Datenblock
    freqs = rfftfreq(len(data), d=1/sample_rate)
    fft_data = rfft(data)

    # Berechnung der Verstärkungsfaktoren für die Ziel-Frequenz
    total_gain = np.ones_like(fft_data)
    gain = weighted_gain(target_frequency, freqs, max_gain)
    total_gain *= gain  # Kombiniere Verstärkungen

    # Anwenden des Gesamt-Verstärkungsprofils
    fft_data *= total_gain

    # Inverse FFT zurück in den Zeitbereich
    filtered_data = irfft(fft_data)
    return filtered_data

# Audio-Callback-Funktion für Echtzeit-Bearbeitung
def audio_callback(indata, outdata, frames, time, status):
    """Verarbeitet die Audio-Daten in Echtzeit."""
    if status:
        print(status, flush=True)

    # Wandle das Audio-Signal in ein 1D-NumPy-Array um
    audio_data = indata[:, 0]

    # Wende den Equalizer auf den aktuellen Block an
    equalized_data = apply_weighted_eq(audio_data, SAMPLE_RATE, TARGET_FREQUENCY, MAX_GAIN)

    # Schreibe die bearbeiteten Daten zum Output-Puffer
    outdata[:, 0] = equalized_data

# Starten des Streams: Verwende das virtuelle Gerät als Input, Lautsprecher als Output
with sd.Stream(device=(input_device, output_device), channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, callback=audio_callback):
    print("Live-Audio-Equalizer läuft... Drücke 'Strg+C' zum Beenden.")
    try:
        sd.sleep(1000000)  # Laufen lassen, bis der Benutzer den Prozess beendet
    except KeyboardInterrupt:
        print("\nLive-Audio-Equalizer beendet.")
