import pandas as pd

# Replace with your actual file name
df = pd.read_csv("../datasets/malicious_phish.csv")

print("Columns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

print("\nType value counts:")
print(df["type"].value_counts())    