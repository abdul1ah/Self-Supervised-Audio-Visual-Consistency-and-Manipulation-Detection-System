import os
import glob
import random
import torch
import torchaudio
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from config import *

class RAVDESSDataset(Dataset):
    def __init__(self, data_dir, frames_per_clip=45, is_train=True):
        """
        Dynamically loads MP4 files, extracts video/audio, and generates synthetic mismatches.
        """
        super().__init__()
        self.frames_per_clip = frames_per_clip

        search_pattern = os.path.join(data_dir, "**", "01-*.mp4")
        all_files = glob.glob(search_pattern, recursive=True)
        
        if len(all_files) == 0:
            raise FileNotFoundError(f"No MP4 files found in {data_dir}. Check your path!")
            
        all_files.sort()
        
        split_idx = int(len(all_files) * 0.8)
        if is_train:
            self.files = all_files[:split_idx]
        else:
            self.files = all_files[split_idx:]
            
        print(f"Loaded {len(self.files)} videos for {'Training' if is_train else 'Validation'}")

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=48000,
            n_mels=128,
            n_fft=2048,
            hop_length=512
        )

    def __len__(self):
        return len(self.files) * 2 

    def __getitem__(self, idx):

        actual_idx = idx % len(self.files)
        video_path = self.files[actual_idx]
        
        label = random.choice([0, 1])
        
        vframes, aframes, info = torchvision.io.read_video(video_path, pts_unit='sec', output_format='TCHW')
        
        video_fps = info.get('video_fps', 30.0)
        audio_fps = info.get('audio_fps', 48000)
        
        total_frames = vframes.shape[0]

        if total_frames <= self.frames_per_clip:
            start_frame = 0
        else:
            start_frame = random.randint(0, total_frames - self.frames_per_clip - 1)
            
        end_frame = start_frame + self.frames_per_clip
        
        visual_clip = vframes[start_frame:end_frame]
        
        visual_clip = visual_clip.float() / 255.0
        visual_clip = F.interpolate(visual_clip, size=(224, 224), mode='bilinear', align_corners=False)
        visual_clip = visual_clip.permute(1, 0, 2, 3) 
        
        clip_duration_sec = self.frames_per_clip / video_fps
        audio_samples_needed = int(clip_duration_sec * audio_fps)
        
        base_audio_start = int((start_frame / video_fps) * audio_fps)
        
        if label == 1:
            audio_start = base_audio_start
            audio_clip = aframes[:, audio_start : audio_start + audio_samples_needed]
            
        else:
            shift_direction = random.choice([-1, 1])
            shift_seconds = random.uniform(0.5, 1.5)
            shift_samples = int(shift_seconds * audio_fps) * shift_direction
            
            audio_start = base_audio_start + shift_samples
            
            if audio_start < 0 or (audio_start + audio_samples_needed) > aframes.shape[1]:

                audio_start = random.randint(0, max(0, aframes.shape[1] - audio_samples_needed))
                
            audio_clip = aframes[:, audio_start : audio_start + audio_samples_needed]

        if audio_clip.shape[1] < audio_samples_needed:
            padding = audio_samples_needed - audio_clip.shape[1]
            audio_clip = F.pad(audio_clip, (0, padding))
            
        if audio_clip.shape[0] > 1:
            audio_clip = torch.mean(audio_clip, dim=0, keepdim=True)

        spectrogram = self.mel_transform(audio_clip)
        spectrogram = torch.log(spectrogram + 1e-9)
        
        spectrogram = spectrogram.unsqueeze(0) 
        spectrogram = F.interpolate(spectrogram, size=(224, 224), mode='bilinear', align_corners=False)
        spectrogram = spectrogram.squeeze(0)

        label_tensor = torch.tensor([label], dtype=torch.float32)

        return visual_clip, spectrogram, label_tensor

def get_dataloader(data_dir, batch_size, shuffle, is_train=True):
    dataset = RAVDESSDataset(data_dir, frames_per_clip=45, is_train=is_train)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=2, 
        pin_memory=True,
        drop_last=True
    )
    return dataloader