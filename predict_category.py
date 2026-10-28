
import os
import pickle

# Putanja do sacuvanog modela
MODEL_PATH = "models/product_classifier.pkl"

# Provera da li model postoji
if not os.path.exists(MODEL_PATH):
    print("Greska: Model nije pronadjen!")
    print("Prvo pokrenite train_model.py")
    exit()

# Ucitavanje modela
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

print("Model je uspesno ucitan!")
print("Automatska klasifikacija proizvoda")
print("-" * 40)

# Interaktivni unos proizvoda
while True:

    product = input(
        "\nUnesite naziv proizvoda (ili 'exit' za kraj): "
    )

    # Izlazak iz programa
    if product.lower() == "exit":
        print("Program je zavrsen.")
        break

    # Provera praznog unosa
    if not product.strip():
        print("Unesite validan naziv proizvoda.")
        continue

    # Predvidjanje kategorije
    prediction = model.predict([product])

    print("Predvidjena kategorija:", prediction[0])
