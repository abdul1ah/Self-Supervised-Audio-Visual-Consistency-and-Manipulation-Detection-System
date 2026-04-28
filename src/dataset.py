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
        for i in range(1, 46): 
            frame_path = os.path.join(FRAMES_DIR, f"{vid_id}_{i:03d}.jpg")
            if not os.path.exists(frame_path):
                break
                
            img = Image.open(frame_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)
            
        while len(frames) < MIN_FRAMES:
            frames.append(frames[-1] if len(frames) > 0 else torch.zeros(3, TARGET_FRAME_SIZE[0], TARGET_FRAME_SIZE[1]))

        visual_tensor = torch.stack(frames, dim=1)

        spec_path = os.path.join(SPECTROGRAMS_DIR, f"{aud_id}.npy")
        full_spectrogram = np.load(spec_path)

        sliced_audio = full_spectrogram[:, :TARGET_AUDIO_STEPS]

        if sliced_audio.shape[1] < TARGET_AUDIO_STEPS:
            padding = TARGET_AUDIO_STEPS - sliced_audio.shape[1]
            sliced_audio = np.pad(sliced_audio, ((0, 0), (0, padding)), mode='constant')

        if shift != 0:
  
            shift_columns = int(shift * 7) 
            
            sliced_audio = np.roll(sliced_audio, shift=shift_columns, axis=1)

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