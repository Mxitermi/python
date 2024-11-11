import os

def main():
    input_file = "C:/daten/python/frequenz_analyse.txt"
    output_file = "C:/daten/python/frequenz_analyse_cut.txt"

    def kürze_auf_vielfaches_von_13(arr):
        # Länge des Arrays so kürzen, dass es ein Vielfaches von 13 ist
        kürzung = len(arr) - (len(arr) % 13)  # Nächstes Vielfaches von 13 kleiner oder gleich der Länge
        return arr[:kürzung]

    try:
        # Liste, um die gekürzten Zeilen zu speichern
        liste = []
        cnt = 0
        # Datei lesen
        with open(input_file, 'r') as file:
            for line in file:
                cnt = cnt + 1
                r = line.split()
                # Kürze die Liste der Floats auf ein Vielfaches von 13
                gekürzte_liste = kürze_auf_vielfaches_von_13(r)
                # Am Ende der gekürzten Liste eine "0" anhängen
                #gekürzte_liste.append("0")
                # Zusammenfügen der gekürzten Liste als String
                new_line = ' '.join(gekürzte_liste)
                liste.append(new_line)

        # In eine neue Datei schreiben
        with open(output_file, 'w') as file:
            for line in liste:
                file.write(line + '\n')

        print("Daten erfolgreich in die Datei geschrieben.")
        
    except FileNotFoundError as e:
        print(f"Datei nicht gefunden: {e}")
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")

if __name__ == "__main__":
    main()
