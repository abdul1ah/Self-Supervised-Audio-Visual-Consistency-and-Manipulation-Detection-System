import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import *

class AudioVisualDataset(Dataset):
    def __init__(self, metadata_path):
        """
        Args:
            metadata_path (str): Path to the metadata.csv file.
        """
        self.metadata = pd.read_csv(metadata_path)
        
        self.transform = transforms.Compose([
            transforms.Resize(TARGET_FRAME_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
            row = self.metadata.iloc[idx]
            vid_id = row['video_id']
            aud_id = row['audio_id']
            label = torch.tensor(row['label'], dtype=torch.float32)
            shift = int(row['shift'])

            frames = []
            for i in range(1, 151): 
                frame_path = os.path.join(FRAMES_DIR, f"{vid_id}_{i:03d}.jpg")
                if not os.path.exists(frame_path):
                    break
                    
                img = Image.open(frame_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                frames.append(img)
                
            while len(frames) < 45:
                frames.append(frames[-1] if len(frames) > 0 else torch.zeros(3, 224, 224))
                
            full_visual_tensor = torch.stack(frames, dim=1)

            spec_path = os.path.join(SPECTROGRAMS_DIR, f"{aud_id}.npy")
            full_spectrogram = np.load(spec_path)

            if shift != 0:
                shift_columns = shift * 10 
                full_spectrogram = np.roll(full_spectrogram, shift=shift_columns, axis=1)

            TARGET_FRAMES = 45 
            TARGET_AUDIO = 100 
            
            total_frames = full_visual_tensor.shape[1]
            total_audio_steps = full_spectrogram.shape[1]
            
            max_frame_start = max(0, total_frames - TARGET_FRAMES)
            
            if hasattr(self, 'is_training') and self.is_training:
                import random
                start_frame = random.randint(0, max_frame_start)
            else:
                start_frame = max_frame_start // 2 
                
            time_ratio = start_frame / total_frames if total_frames > 0 else 0
            start_audio = int(time_ratio * total_audio_steps)
            
            visual_tensor = full_visual_tensor[:, start_frame : start_frame + TARGET_FRAMES, :, :]
            
            sliced_audio = full_spectrogram[:, start_audio : start_audio + TARGET_AUDIO]
            
            if sliced_audio.shape[1] < TARGET_AUDIO:
                padding = TARGET_AUDIO - sliced_audio.shape[1]
                sliced_audio = np.pad(sliced_audio, ((0, 0), (0, padding)), mode='constant')
                
            audio_tensor = torch.tensor(sliced_audio, dtype=torch.float32).unsqueeze(0)

            return visual_tensor, audio_tensor, label

def get_dataloader(csv_path, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True):

    dataset = AudioVisualDataset(csv_path)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader