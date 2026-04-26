import pandas as pd
import os
import random
import glob

# minimum frames required for a video to be included in the dataset
MIN_FRAMES = 5   
random.seed(42)

# load original dataset
df = pd.read_csv("filtered_vggsound.csv")

# filter to only include videos with valid spectrograms and enough frames
valid_ids = []

for file in os.listdir("spectrograms"):
    vid = file.replace(".npy", "")

    spec_path = f"spectrograms/{vid}.npy"
    frame_files = glob.glob(f"frames/{vid}_*.jpg")

    if os.path.exists(spec_path) and len(frame_files) >= MIN_FRAMES:
        valid_ids.append(vid)

print(f"Valid samples: {len(valid_ids)}")

# filter original dataframe to only include valid video IDs
df = df[df['YouTube_ID'].isin(valid_ids)]

video_ids = df['YouTube_ID'].tolist()
labels_map = dict(zip(df['YouTube_ID'], df['label']))

# generate metadata for audio-visual pairing and labeling
metadata = []

for vid in video_ids:

    # positive pair (same video, same class)
    metadata.append({
        "video_id": vid,
        "audio_id": vid,
        "label": 1,
        "type": "positive",
        "shift": 0
    })

    # cross-video negative (different video, different class)
    neg_pool = [v for v in video_ids if v != vid]
    neg_vid = random.choice(neg_pool)

    metadata.append({
        "video_id": vid,
        "audio_id": neg_vid,
        "label": 0,
        "type": "cross_video",
        "shift": 0
    })

    # hard negative (same class, different video)
    same_class = df[df['label'] == labels_map[vid]]['YouTube_ID'].tolist()
    same_class = [v for v in same_class if v != vid]

    if len(same_class) > 0:
        hard_vid = random.choice(same_class)

        metadata.append({
            "video_id": vid,
            "audio_id": hard_vid,
            "label": 0,
            "type": "hard_negative",
            "shift": 0
        })

    # temporal shift negative (same video, different time)
    metadata.append({
        "video_id": vid,
        "audio_id": vid,
        "label": 0,
        "type": "temporal_shift",
        "shift": random.choice([-2, -1, 1, 2])
    })

## save metadata to csv
metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv("metadata.csv", index=False)

print("\nMetadata created!")
print(metadata_df['type'].value_counts())
print(f"Total samples: {len(metadata_df)}")