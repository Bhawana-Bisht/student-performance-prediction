import pandas as pd

df = pd.read_csv("data/raw/students.csv")

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Feature engineering
df["study_efficiency"] = df["study_hours"] / df["screen_time"]
df["support_score"] = df["school_support"] + df["parent_education"]

df.to_csv("data/processed/cleaned_data.csv", index=False)

print("Preprocessing completed")
