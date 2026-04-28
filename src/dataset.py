import os
import glob
import random
import torch
import torchaudio
import torchvision
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from config import *

class RAVDESSDataset(Dataset):
    def __init__(self, preprocessed_dir, is_train=True):
        self.preprocessed_dir = preprocessed_dir
        all_folders = glob.glob(os.path.join(self.preprocessed_dir, "*"))
        
        train_folders = []
        val_folders = []
        
        for folder in all_folders:
            folder_name = os.path.basename(folder)
            actor_id = int(folder_name.split("-")[-1])
            
            if actor_id <= 19:
                train_folders.append(folder)
            else:
                val_folders.append(folder)
                
        self.folders = train_folders if is_train else val_folders
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=AUDIO_SAMPLE_RATE, n_mels=NUM_MEL_BINS, n_fft=2048, hop_length=512
        )

    def __len__(self):

        return len(self.folders) * 2

    def __getitem__(self, idx):
        folder = self.folders[idx % len(self.folders)]
        label = random.choice([0, 1])

        video = np.load(os.path.join(folder, "video.npy"))
        video = torch.from_numpy(video).float() / 255.0
        video = video.permute(1, 0, 2, 3)

        audio = torch.from_numpy(np.load(os.path.join(folder, "audio.npy")))

        if label == 0:
            shift = random.randint(16000, 32000)
            audio = torch.roll(audio, shifts=shift, dims=1)

        spectrogram = self.mel_transform(audio)
        spectrogram = torch.log(spectrogram + 1e-9)
        spectrogram = F.interpolate(spectrogram.unsqueeze(0), size=(224, 224)).squeeze(0)

        return video, spectrogram, torch.tensor([label], dtype=torch.float32)

def get_dataloader(data_dir, batch_size, shuffle, is_train=True):
    dataset = RAVDESSDataset(data_dir, is_train=is_train)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        drop_last=True
    )
    return dataloader