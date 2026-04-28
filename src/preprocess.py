import os
import glob
import torch
import torchvision
import numpy as np
from tqdm import tqdm
import torchvision.transforms.functional as F
from config import *

TEMP_DATA_DIR = "/kaggle/temp/ravdess_preprocessed"
os.makedirs(TEMP_DATA_DIR, exist_ok=True)

def preprocess_dataset():

    search_pattern = os.path.join(RAVDESS_DIR, "**", "01-*.mp4")
    video_files = glob.glob(search_pattern, recursive=True)
    print(f"Found {len(video_files)} videos to process.")

    for video_path in tqdm(video_files, desc="Pre-processing"):
        try:

            video_id = os.path.basename(video_path).replace(".mp4", "")
            video_out_dir = os.path.join(TEMP_DATA_DIR, video_id)
            os.makedirs(video_out_dir, exist_ok=True)

            vframes, aframes, info = torchvision.io.read_video(video_path, pts_unit='sec', output_format='TCHW')
            
            total_v = vframes.shape[0]
            if total_v < 45: continue
            start_v = (total_v - 45) // 2
            v_clip = vframes[start_v : start_v + 45]

            v_clip = torchvision.transforms.Resize((224, 224))(v_clip)
            
            np.save(os.path.join(video_out_dir, "video.npy"), v_clip.numpy().astype(np.uint8))

            v_fps = info.get('video_fps', 30.0)
            a_fps = info.get('audio_fps', 48000)
            
            a_start = int((start_v / v_fps) * a_fps)
            a_end = int(((start_v + 45) / v_fps) * a_fps)
            a_clip = aframes[:, a_start:a_end]
            
            if a_clip.shape[0] > 1:
                a_clip = torch.mean(a_clip, dim=0, keepdim=True)

            np.save(os.path.join(video_out_dir, "audio.npy"), a_clip.numpy())

        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue

if __name__ == "__main__":
    preprocess_dataset()