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
        for i in range(1, MIN_FRAMES + 1): 
            frame_path = os.path.join(FRAMES_DIR, f"{vid_id}_{i:03d}.jpg")
            
            if not os.path.exists(frame_path) and len(frames) > 0:
                frames.append(frames[-1])
                continue
                
            img = Image.open(frame_path).convert('RGB')
            img_tensor = self.transform(img)
            frames.append(img_tensor)
        
        visual_tensor = torch.stack(frames, dim=1) 

        spec_path = os.path.join(SPECTROGRAMS_DIR, f"{aud_id}.npy")
        spectrogram = np.load(spec_path)

        if shift != 0:
            shift_columns = shift * 10 
            spectrogram = np.roll(spectrogram, shift=shift_columns, axis=1)

        TARGET_AUDIO_LENGTH = 300 
        current_length = spectrogram.shape[1]

        if current_length < TARGET_AUDIO_LENGTH:
            padding = TARGET_AUDIO_LENGTH - current_length
            spectrogram = np.pad(spectrogram, ((0, 0), (0, padding)), mode='constant')
        elif current_length > TARGET_AUDIO_LENGTH:
            spectrogram = spectrogram[:, :TARGET_AUDIO_LENGTH]


        audio_tensor = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)

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