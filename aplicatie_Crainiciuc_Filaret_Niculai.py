import joblib

def predict_from_file():
    # Încarcă modelul și scalerul
    model = joblib.load("asteroid_model.pkl")
    scaler = joblib.load("scaler.pkl")

    # Citire obiecte multiple din fișier (unul pe fiecare linie)
    with open("sample_input.txt", "r") as f:
        lines = f.readlines()

    # Procesează fiecare obiect separat
    for i,line in enumerate(lines):
        # Ignoră liniile goale
        if not line.strip():
            continue
            
        # Extrage și prelucrează caracteristicile
        raw_input = line.strip().split(",")  # CSV format
        
        try:
            features = [float(val) for val in raw_input]
            scaled = scaler.transform([features])
            prediction = model.predict(scaled)
            
            print(f"{i:2}. ", end="")
            # Afișează rezultatul pentru fiecare obiect
            print(f"Periculos" if prediction[0] == 1 else 'Nepericulos')
        except ValueError as e:
            print(f"Eroare la procesarea obiectului: {e}")
        except Exception as e:
            print(f"Eroare neașteptată la obiectul: {e}")