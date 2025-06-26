import pandas as pd
import numpy as np

np.random.seed(42)
n = 500

data = {
    "Age": np.random.randint(20, 70, n),
    "Gender": np.random.choice(["male", "female"], n),
    "Smoking": np.random.choice(["yes", "no"], n, p=[0.3, 0.7]),
    "Alcohol": np.random.choice(["yes", "no"], n, p=[0.4, 0.6]),
    "FamilyHistory": np.random.choice(["yes", "no"], n, p=[0.3, 0.7]),
    "Occupation": np.random.choice(["office", "manual"], n),
    "Diet": np.random.choice(["poor", "average", "good"], n, p=[0.3, 0.4, 0.3]),
    "ExerciseFreq": np.random.choice(["low", "medium", "high"], n, p=[0.3, 0.4, 0.3]),
    "HeightCm": np.random.randint(150, 190, n),
    "WeightKg": np.random.randint(50, 100, n),
}

df = pd.DataFrame(data)
df["BMI"] = df["WeightKg"] / ((df["HeightCm"] / 100) ** 2)

df["TumorRisk"] = (
    ((df["Smoking"] == "yes") & (df["Alcohol"] == "yes") & (df["FamilyHistory"] == "yes")).astype(int)
    | ((df["BMI"] > 28) & (df["Diet"] == "poor")).astype(int)
).astype(int)

df = df.drop(columns=["BMI"])
df.to_csv("synthetic_lifestyle_data.csv", index=False)

print("✅ Synthetic lifestyle data created with 500 rows.")
