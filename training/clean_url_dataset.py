import pandas as pd

# Load dataset
df = pd.read_csv("../datasets/malicious_phish.csv")

# Keep only phishing and benign
df = df[df["type"].isin(["phishing", "benign"])]

# Convert to binary label
df["label"] = df["type"].apply(lambda x: 1 if x == "phishing" else 0)

# Keep only required columns
df = df[["url", "label"]]

# Remove duplicates and nulls
df = df.drop_duplicates().dropna()

print("Before balancing:")
print(df["label"].value_counts())

# Balance dataset
phishing_df = df[df["label"] == 1]
benign_df = df[df["label"] == 0].sample(len(phishing_df), random_state=42)

balanced_df = pd.concat([phishing_df, benign_df])

# Shuffle
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nAfter balancing:")
print(balanced_df["label"].value_counts())

# Save cleaned dataset
balanced_df.to_csv("../datasets/url_data_clean.csv", index=False)

print("\nFinal dataset size:", len(balanced_df))