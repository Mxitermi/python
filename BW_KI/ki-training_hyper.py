import torch
import numpy as np
import torch.nn as nn
import os
from sklearn.model_selection import ParameterGrid  # Für das Grid Search

# Trainingsfunktion
def train(D, learning_rate, hidden_size, batch_size):
    len_train = int(np.size(D, 0) * 0.8)
    train, test = D[:len_train], D[len_train:]

    # Hyper-Parameter
    n_steps = 3000  # Epochenzahl, kann bei Bedarf erhöht werden
    input_size = 13  # Anzahl der Eingabewerte pro Datensatz
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
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)  # Letzte Schicht für binäre Klassifikation
            )

        def forward(self, x):
            return self.layers(x)

    model = MLP(input_size, hidden_size)

    # Verlustfunktion und Optimierer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Modell trainieren
    for e in range(n_steps):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Testen und Accuracy berechnen
    with torch.no_grad():
        outputs_test = model(X_test)
        pred_y_test = torch.sigmoid(outputs_test).view(-1) > 0.5
        accuracy_test = (pred_y_test.float() == y_test.view(-1)).float().mean().item()

    return accuracy_test, loss.item()

# Grid Search Funktion
def grid_search(D):
    # Hyperparameter-Räume definieren
    param_grid = {
        'learning_rate': [0.01, 0.001, 0.0001],
        'hidden_size': [64, 128, 256],
        'batch_size': [32, 64, 128]
    }

    best_accuracy = 0
    best_params = None

    for params in ParameterGrid(param_grid):
        learning_rate = params['learning_rate']
        hidden_size = params['hidden_size']
        batch_size = params['batch_size']

        print(f"Training with lr={learning_rate}, hidden_size={hidden_size}, batch_size={batch_size}")
        
        accuracy, loss = train(D, learning_rate, hidden_size, batch_size)
        print(f"Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
        
        # Update best params if we found better accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params

    print(f"Beste Hyperparameter: {best_params} mit Accuracy: {best_accuracy:.4f}")

def data_structure(data):
    out = np.array(data, dtype=np.float32)
    return out

def main():
    data = []
    filename = "data_real.txt"
    try:
        with open(os.path.join(os.path.dirname(__file__), filename), "r", encoding="utf-8") as file:
            for line in file:
                values = list(map(float, line.strip().split()))
                label = values[-1]
                data_points = values[:-1]

                for i in range(0, len(data_points), 13):
                    batch = data_points[i:i+13]
                    if len(batch) == 13:
                        data.append(batch + [label])

        # Gesamtes Dataset einmal mischen
        np.random.shuffle(data)

    except IOError:
        print("Datei existiert nicht. Stellen Sie sicher, dass die Datei vorhanden ist und im richtigen Verzeichnis liegt.")
        return

    # Grid Search ausführen
    grid_search(data_structure(data))

if __name__ == "__main__":
    main()
