import torch
import numpy as np
import torch.nn as nn
import os

def train(D):
    len_train = int(np.size(D, 0) * 0.8)
    train, test = D[:len_train], D[len_train:]

    # Hyper-Parameter
    n_steps = 10000
    learning_rate = 0.003
    input_size = 13  # Die Anzahl der Werte pro Datensatz
    output_size = 1

    # Trainings-Daten vorbereiten
    X = train[:, :-1].astype(np.float32)
    y = train[:, -1].astype(np.float32).reshape(-1, 1)
    X_train = torch.from_numpy(X)
    y_train = torch.from_numpy(y)
    
    # Normalisierung
    feature_means = torch.mean(X_train, dim=0)
    feature_stds = torch.std(X_train, dim=0, unbiased=False)
    X_train = (X_train - feature_means) / feature_stds

    # Test-Daten vorbereiten
    X = test[:, :-1].astype(np.float32)
    y = test[:, -1].astype(np.float32).reshape(-1, 1)
    X_test = torch.from_numpy(X)
    y_test = torch.from_numpy(y)
    X_test = (X_test - feature_means) / feature_stds

    # Modell definieren
    class MLP(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, 128),  # Optional: größere erste Schicht
                nn.ReLU(),
                nn.Linear(128, 64),  # Optional: zweite größere Schicht
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),  # Batch-Normalisierung für stabileres Training
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)  # Letzte Schicht für binäre Klassifikation
            )

        def forward(self, x):
            return self.layers(x)


    model = MLP(input_size)

    # Verlustfunktion
    criterion = nn.BCEWithLogitsLoss()

    # Optimierer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Modell trainieren
    smallest = 100
    epoch = 0
    for e in range(n_steps):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 100 == 0:
            with torch.no_grad():
                outputs_train = model(X_train)
                pred_y_train = torch.sigmoid(outputs_train).view(-1) > 0.5
                accuracy_train = (pred_y_train.float() == y_train.view(-1)).float().mean().item()

                outputs_test = model(X_test)
                pred_y_test = torch.sigmoid(outputs_test).view(-1) > 0.5
                accuracy_test = (pred_y_test.float() == y_test.view(-1)).float().mean().item()

                print(f'Epoch {e}, Loss: {loss:.4f}, Acc train: {accuracy_train:.4f}, Acc test: {accuracy_test:.4f}')
                if(smallest > loss):
                    smallest = loss
                    epoch = e
    print(f"Bestes Ergebnis: {smallest:.6f} : {epoch}")
    return model

def data_structure(data):
    out = np.array(data, dtype=np.float32)
    return out

def main():
    data = []
    filename = "data_real.txt"
    try:
        with open(os.path.join(os.path.dirname(__file__), filename), "r", encoding="utf-8") as file:
            for line in file:
                # Werte aus der Zeile extrahieren und in Float umwandeln
                values = list(map(float, line.strip().split()))
                
                # Letztes Element als Label (0 oder 1) extrahieren
                label = values[-1]
                # Alle anderen Werte als Datenpunkte extrahieren
                data_points = values[:-1]

                # Sicherstellen, dass die Anzahl der Datenpunkte ein Vielfaches von 13 ist
                for i in range(0, len(data_points), 13):
                    # Ein Paket von 13 Werten erstellen
                    batch = data_points[i:i+13]
                    if len(batch) == 13:
                        # Paket mit Label hinzufügen
                        data.append(batch + [label])

        # **Gesamtes Dataset einmal mischen**
        np.random.shuffle(data)

    except IOError:
        print("Datei existiert nicht. Stellen Sie sicher, dass die Datei vorhanden ist und im richtigen Verzeichnis liegt.")
        return

    # Trainiere das Modell mit den aufbereiteten Daten
    model = train(data_structure(data))
    torch.save(model.state_dict(), "model_real.pt")

if __name__ == "__main__":
    main()