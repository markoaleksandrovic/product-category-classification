import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# 1. Ucitavanje podataka

print("Ucitavanje podataka...")

df = pd.read_csv("products.csv")
df.columns = df.columns.str.strip()

print("Broj originalnih proizvoda:", len(df))


# 2. Ciscenje podataka

df = df[["Product Title", "Category Label"]].copy()

df = df.dropna(
    subset=["Product Title", "Category Label"]
)

df["Product Title"] = (
    df["Product Title"]
    .astype(str)
    .str.lower()
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

df["Category Label"] = (
    df["Category Label"]
    .astype(str)
    .str.strip()
)

df = df[
    (df["Product Title"] != "") &
    (df["Category Label"] != "")
].copy()

# Uklanjanje dupliranih parova
df = df.drop_duplicates(
    subset=["Product Title", "Category Label"]
)

# Uklanjanje naziva sa konfliktnim kategorijama
conflicts = (
    df.groupby("Product Title")["Category Label"]
    .nunique()
)

conflicting_titles = conflicts[conflicts > 1].index

df = df[
    ~df["Product Title"].isin(conflicting_titles)
].copy()


# 3. Standardizacija kategorija

category_mapping = {
    "CPU": "CPUs",
    "Mobile Phone": "Mobile Phones",
    "fridge": "Fridges"
}

df["Category Label"] = (
    df["Category Label"].replace(category_mapping)
)

# Zavrsna provera
assert df["Product Title"].duplicated().sum() == 0
assert df.isnull().sum().sum() == 0

print("Broj ociscenih proizvoda:", len(df))
print("Broj kategorija:", df["Category Label"].nunique())


# 4. Podela podataka

X = df["Product Title"]
y = df["Category Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 5. Kreiranje ML Pipeline-a

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=20000,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])


# 6. Treniranje modela

print("Treniranje modela...")

model.fit(X_train, y_train)


# 7. Evaluacija modela

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))


# 8. Cuvanje modela

os.makedirs("models", exist_ok=True)

model_path = "models/product_classifier.pkl"

with open(model_path, "wb") as file:
    pickle.dump(model, file)

print("\nModel je sacuvan:", model_path)
