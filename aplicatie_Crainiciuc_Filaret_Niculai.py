import joblib

# Încarcă modelul și scalerul
model = joblib.load("asteroid_model.pkl")
scaler = joblib.load("scaler.pkl")

# Exemplu: 6 caracteristici -> citire din fișier
with open("sample_input.txt", "r") as f:
    raw_input = f.read().strip().split(",")  # CSV format

features = [float(val) for val in raw_input]
scaled = scaler.transform([features])
prediction = model.predict(scaled)

print("Este periculos!" if prediction[0] == 1 else "Nu este periculos.")
