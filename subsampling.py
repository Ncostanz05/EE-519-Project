import pandas as pd
import os
import shutil

CSV_PATH = "commonvoice-v24_en-AU/commonvoice-v24_en-AU.csv"
AUDIO_DIR = "commonvoice-v24_en-AU/audio_files"
OUTPUT_DIR = "cv_au_subset"
SAMPLES_PER_GROUP = 1000

os.makedirs(f"{OUTPUT_DIR}/audio", exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Remove empty ages
df = df[df["age"].notna()]

# Drop extremely small group
df = df[df["age"] != "eighties"]

balanced = []

for age_group in df["age"].unique():
    subset = df[df["age"] == age_group]
    subset = subset.sample(
        min(SAMPLES_PER_GROUP, len(subset)),
        random_state=42
    )
    balanced.append(subset)

balanced_df = pd.concat(balanced)

# Copy selected audio files
for _, row in balanced_df.iterrows():
    src = os.path.join(AUDIO_DIR, row["path"])
    dst = os.path.join(OUTPUT_DIR, "audio", row["path"])
    shutil.copy(src, dst)

balanced_df.to_csv(f"{OUTPUT_DIR}/metadata.csv", index=False)

print("Subset created successfully.")