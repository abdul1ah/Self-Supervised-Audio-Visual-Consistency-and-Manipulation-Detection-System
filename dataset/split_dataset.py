import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load the master metadata
df = pd.read_csv("metadata.csv")

# 2. Get unique video IDs to prevent data leakage!
unique_vids = df['video_id'].unique()
print(f"Total unique videos: {len(unique_vids)}")

# 3. Split IDs: 80% Train, 20% Temp
train_ids, temp_ids = train_test_split(unique_vids, test_size=0.2, random_state=42)

# 4. Split Temp: 50% Val, 50% Test (which equals 10% of total each)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

print(f"Train videos: {len(train_ids)} | Val videos: {len(val_ids)} | Test videos: {len(test_ids)}")

# 5. Filter the original dataframe based on these isolated IDs
train_df = df[df['video_id'].isin(train_ids)]
val_df = df[df['video_id'].isin(val_ids)]
test_df = df[df['video_id'].isin(test_ids)]

# 6. Save the new CSVs
train_df.to_csv("train_metadata.csv", index=False)
val_df.to_csv("val_metadata.csv", index=False)
test_df.to_csv("test_metadata.csv", index=False)

print("\nDataset successfully split and saved")