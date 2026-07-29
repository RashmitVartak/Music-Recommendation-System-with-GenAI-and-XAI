import pandas as pd

df = pd.read_csv("datasets/processed/merged_dataset.csv")

print("=" * 60)
print("MERGED DATASET SUMMARY")
print("=" * 60)

print(f"Shape: {df.shape}")

print("\nMissing Values:")
print(df[
    [
        "album_name",
        "genres",
        "artist_popularity",
        "artist_followers",
        "release_date",
        "year"
    ]
].isna().sum())

print("\nDuplicate IDs:")
print(df["id"].duplicated().sum())

print("\nDuplicate Song + Artist:")
print(df.duplicated(["name", "artists"]).sum())

print("\nFirst 5 Rows:")
print(df.head())