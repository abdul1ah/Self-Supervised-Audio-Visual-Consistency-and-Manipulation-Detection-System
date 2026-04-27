import pandas as pd
import os
import subprocess
import librosa
import numpy as np

df = pd.read_csv("filtered_vggsound.csv")

CLIP_DURATION = 10

success, failed, skipped = 0, 0, 0
total = len(df)

os.makedirs("audio", exist_ok=True)
os.makedirs("spectrograms", exist_ok=True)
os.makedirs("frames", exist_ok=True)

for _, row in df.iterrows():
    vid = row['YouTube_ID']
    start = int(row['start_seconds'])
    end = start + CLIP_DURATION

    url = f"https://www.youtube.com/watch?v={vid}"

    video_path = f"{vid}.mp4"
    audio_path = f"audio/{vid}.wav"
    spec_path = f"spectrograms/{vid}.npy"

    # skip if already processed
    if os.path.exists(spec_path):
        print(f"[SKIPPED] {vid}")
        skipped += 1
        continue

    try:
            # download segment
            subprocess.run([
                "yt-dlp",
                "--download-sections", f"*{start}-{end}",
                "-f", "bestvideo[height<=360]+bestaudio/best",
                "--merge-output-format", "mp4",
                "-o", video_path,
                url
            ], check=True, timeout=180)  

            # extract audio
            subprocess.run([
                "ffmpeg", "-y",
                "-i", video_path,
                "-q:a", "0",
                "-map", "a",
                audio_path
            ], check=True, timeout=120)   

            # extract frames
            subprocess.run([
                "ffmpeg", "-y",
                "-i", video_path,
                "-vf", "fps=15",
                f"frames/{vid}_%03d.jpg"
            ], check=True, timeout=120)   

            # generate spectrogram
            y, sr = librosa.load(audio_path, sr=16000)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            log_mel_spec = librosa.power_to_db(mel_spec)

            np.save(spec_path, log_mel_spec)

            # delete video
            if os.path.exists(video_path):
                os.remove(video_path)

            success += 1
            print(f"[OK] {vid}")

    except Exception as e:
        print(f"[FAILED] {vid} | {e}")
        failed += 1

print("\n----------SUMMARY-----------")
print(f"Total entries     : {total}")
print(f"Successful        : {success}")
print(f"Failed            : {failed}")
print(f"Skipped (cached)  : {skipped}")
print(f"Processed (new)   : {success + failed}")