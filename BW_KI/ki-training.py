import torch
import numpy as np
import torch.nn as nn
import os

def train(D):
    np.random.shuffle(D)
    len_train = int(np.size(D, 0) * 0.8)
    train, test = D[:len_train], D[len_train:]

    # Hyper-Parameter
    n_steps = 400
    learning_rate = 0.005
    input_size = 13  # Die Anzahl der Werte pro Datensatz
    output_size = 1
    ridge_lambda = 0.002  # Ridge Regularisierung (L2 penalty)

    # Trainings-Daten vorbereiten
    X = train[:, :-1].astype(np.float32)
    y = train[:, -1].astype(np.float32).reshape(-1, 1)
    X_train = torch.from_numpy(X)
    y_train = torch.from_numpy(y)


    # Überprüfen und das Label auf 0 setzen, wenn alle Merkmale 0 sind
    zero_rows = (X_train == 0).all(dim=1)  # Finde alle Zeilen, bei denen alle Merkmale 0 sind
    y_train[zero_rows] = 0  # Setze das Label für diese Zeilen auf 0

    # Test-Daten vorbereiten
    X = test[:, :-1].astype(np.float32)
    y = test[:, -1].astype(np.float32).reshape(-1, 1)
    X_test = torch.from_numpy(X)
    y_test = torch.from_numpy(y)

    # Überprüfen und das Label auf 0 setzen, wenn alle Merkmale 0 sind
    zero_rows_test = (X_test == 0).all(dim=1)  # Finde alle Zeilen, bei denen alle Merkmale 0 sind
    y_test[zero_rows_test] = 0  # Setze das Label für diese Zeilen auf 0


    # Modell definieren
    class MLP(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
             nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)  # Ausgabe für binäre Klassifikation
        )

        def forward(self, x):
            return self.layers(x)

    model = MLP(input_size)
    criterion = nn.BCEWithLogitsLoss()

    # Optimierer mit L2-Regularisierung (Ridge)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=ridge_lambda)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Modell trainieren
    smallest = 100
    epoch = 0
    for e in range(n_steps):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Lernraten-Scheduler auf den Verlust anwenden
        scheduler.step(loss)
        
        if e % 100 == 0:
            with torch.no_grad():
                outputs_train = model(X_train)
                pred_y_train = torch.sigmoid(outputs_train).view(-1) > 0.5
                accuracy_train = (pred_y_train.float() == y_train.view(-1)).float().mean().item()

                outputs_test = model(X_test)
                pred_y_test = torch.sigmoid(outputs_test).view(-1) > 0.5
                accuracy_test = (pred_y_test.float() == y_test.view(-1)).float().mean().item()

                print(f'Epoch {e}, Loss: {loss:.4f}, Acc train: {accuracy_train:.4f}, Acc test: {accuracy_test:.4f}')
                
                if smallest > loss:
                    smallest = loss
                    epoch = e
                    
    print(f"Bestes Ergebnis: {smallest:.6f} : {epoch}")
    return model

def data_structure(data):
    out = np.array(data, dtype=np.float32)
    return out

def main():
    data = []
    filename = "../dataset_final.txt"
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

    except IOError:
        print("Datei existiert nicht. Stellen Sie sicher, dass die Datei vorhanden ist und im richtigen Verzeichnis liegt.")
        return

    # Trainiere das Modell mit den aufbereiteten Daten
    model = train(data_structure(data))
    torch.save(model.state_dict(), "model_real.pt")

if __name__ == "__main__":
    main()
