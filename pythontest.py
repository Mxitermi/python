import sounddevice as sd

# Beispiel: Du verwendest ein spezifisches Gerät für die Audioausgabe
output_device_id = 3  # Angenommen, du hast ein Gerät mit ID 3 ausgewählt

# Vergleiche es mit dem Standard-Ausgabegerät
standard_output_device = sd.default.device[1]
if output_device_id == standard_output_device:
    print("Das verwendete Gerät ist das Standard-Ausgabegerät.")
else:
    print("Das verwendete Gerät ist NICHT das Standard-Ausgabegerät.")
