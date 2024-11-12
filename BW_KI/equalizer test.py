import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# Definition des MLP Modells
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)  # Ausgabe für binäre Klassifikation
        )

    def forward(self, x):
        return self.layers(x)

# Funktion um Eingabewerte zu setzen und die Vorhersage zu machen
def test_model(model, input_values):
    # Umwandeln der Eingabewerte in ein Tensor-Objekt
    input_tensor = torch.tensor(input_values, dtype=torch.float32).unsqueeze(0)  # unsqueeze fügt eine Batch-Dimension hinzu
    
    # Vorhersage des Modells
    with torch.no_grad():  # Kein Gradientenberechnen, da wir nur testen
        output = model(input_tensor)
    
    return output

# Modell initialisieren
input_size = 13  # Anzahl der Eingabewerte
output_size = 1  # Eine Ausgabe (binär, z.B. 0 oder 1)
model = MLP(input_size, output_size)

# Beispiel-Eingabewerte (alle 1.0)
input_values = [24.7, 28.0, 10.3, 16.7, 17.5, 29.3, 12.9, 20.1, 8.2, 10.1, 24.5, 14.5, 20.6]

# Modell testen
output = test_model(model, input_values)
x = 1/(1+np.exp(-output))

# Ausgabe des Ergebnisses
print(f"Das Modell hat folgendes Ergebnis für die Eingabewerte {input_values} geliefert: {x.item()}")
