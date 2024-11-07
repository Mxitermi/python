import torch
import numpy as np
import torch.nn as nn

# Vorhersage-Funktion
def pred(input, model):
    with torch.no_grad():
        pred = model(input)
    pred = pred  # Optional: falls du diese Transformation wünschst
    pred = torch.sigmoid(pred)
    # Runden auf die dritte Nachkommastelle
    rounded_pred = torch.round(pred * 1000) / 1000
    return rounded_pred

# Modell laden
def load_model():
    output_size = 1
    input_size = 13  # Jetzt 13 Werte als Eingabe

    # Einfaches MLP-Netzwerk
    class MLP(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, 64),  # Erste Schicht: 13 -> 64
                nn.ReLU(),
                nn.Linear(64, 32),  # Zweite Schicht: 64 -> 32
                nn.ReLU(),
                nn.Linear(32, 16),  # Dritte Schicht: 32 -> 16
                nn.ReLU(),
                nn.Linear(16, output_size)  # Letzte Schicht: 16 -> 1 (für 0 oder 1 als Ausgabe)
            )

        def forward(self, x):
            out = self.layers(x)
            return out

    model = MLP(input_size)
    # Lade die Modellgewichte
    model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu'), weights_only=True))
    return model

# Funktion zum Einlesen der Datei und Vorhersage für jedes 13er-Paket
def process_file(filename, model):
    results = []  # Hier speichern wir die Ergebnisse
    with open(filename, 'r') as file:
        # Datei einlesen und Werte extrahieren
        lines = file.readlines()

        for line in lines:
            # Splitte die Zeile durch Leerzeichen und konvertiere in Float
            values = list(map(float, line.strip().split()))
            
            # Letztes Element ist das Label
            label = values[-1]
            # Die ersten 13 Werte sind die Eingabedaten
            input_values = values[:-1]

            # Verarbeite die Werte in 13er Paketen
            for i in range(0, len(input_values), 13):
                batch = input_values[i:i+13]  # Nimm ein Paket von 13 Werten
                if len(batch) < 13:
                    break  # Beende, wenn weniger als 13 Werte im letzten Paket

                # Konvertiere die Daten in ein Tensor-Format
                input_tensor = torch.tensor([batch], dtype=torch.float32)
                
                # Mache eine Vorhersage
                prediction = pred(input_tensor, model)
                
                # Ergebnis (0 oder 1) runden und speichern
                rounded_prediction = torch.round(prediction).item()
                results.append(rounded_prediction)

    return results

# Hauptprogramm
if __name__ == "__main__":
    model = load_model()  # Lade das trainierte Modell
    model.eval()  # Setze das Modell in den Evaluierungsmodus

    # Datei mit den Frequenzdaten
    filename = 'frequenz_analyse.txt'
    
    # Verarbeite die Datei und erhalte die Ergebnisse
    predictions = process_file(filename, model)
    frequencies = ['250Hz', "300Hz", "400Hz"]  # Beispielhafte Frequenzen

    # Parameter
    group_size = 13  # Jede Frequenz hat 13 Werte

    # Über die Frequenzen iterieren
    for freq_idx, freq in enumerate(frequencies):
        rueckkopplung_erkannt = False
        print(f"Frequenz: {freq}")
        
        # Für jede Frequenz die 13 entsprechenden Werte extrahieren
        start_idx = freq_idx * group_size
        end_idx = start_idx + group_size
        prediction_slice = predictions[start_idx:end_idx]
        # Durch die Werte für diese Frequenz iterieren
        for idx, wert in enumerate(prediction_slice):
            if wert > 0.5:
                rueckkopplung_erkannt = True
                zeitpunkt = ((start_idx + idx) * 0.1) + 0.1  # Zeit in Sekunden basierend auf dem Index
                print(f"  Rückkopplung erkannt bei {zeitpunkt:.1f} Sekunden.")
            else:
                if rueckkopplung_erkannt:
                    print(f"  Keine Rückkopplung erkannt bei {zeitpunkt:.1f} Sekunden.")
        
        # Wenn für diese Frequenz keine Rückkopplung erkannt wurde
        if not rueckkopplung_erkannt:
            print("  Keine Rückkopplung erkannt.")

