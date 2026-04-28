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
    def __init__(self, preprocessed_dir, is_train=True):
        self.preprocessed_dir = "/kaggle/temp/ravdess_preprocessed"
        self.folders = sorted(glob.glob(os.path.join(self.preprocessed_dir, "*")))
        
        split = int(len(self.folders) * 0.8)
        self.folders = self.folders[:split] if is_train else self.folders[split:]
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=48000, n_mels=128, n_fft=2048, hop_length=512
        )

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