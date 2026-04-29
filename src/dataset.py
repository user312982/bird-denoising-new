import os
import glob
import librosa
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

class BirdAudioDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, config):
        self.noisy_files = sorted(glob.glob(os.path.join(noisy_dir, '*.wav')))
        self.clean_files = sorted(glob.glob(os.path.join(clean_dir, '*.wav')))
        self.config = config
        
        if len(self.noisy_files) == 0:
            print(f"Warning: No .wav files found in {noisy_dir}")
        elif len(self.noisy_files) != len(self.clean_files):
            print("Warning: Number of noisy and clean files do not match!")

    def __len__(self):
        return len(self.noisy_files)

    def _pad_or_crop_spectrogram(self, spec):
        target_size = self.config.IMAGE_SIZE
        h, w = spec.shape
        
        # Pad or crop height (frequency)
        if h < target_size:
            spec = np.pad(spec, ((0, target_size - h), (0, 0)), mode='constant')
        else:
            spec = spec[:target_size, :]
            
        # Pad or crop width (time)
        if w < target_size:
            spec = np.pad(spec, ((0, 0), (0, target_size - w)), mode='constant')
        else:
            spec = spec[:, :target_size]
            
        return spec

    def _robust_load(self, path):
        try:
            waveform, sr = torchaudio.load(path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sr != self.config.SR:
                waveform = torchaudio.functional.resample(waveform, sr, self.config.SR)
            return waveform.squeeze(0).numpy()
        except Exception as e:
            return None

    def __getitem__(self, idx):
        noisy_path = self.noisy_files[idx]
        clean_path = self.clean_files[idx]
        
        # Load audio safely
        noisy_audio = self._robust_load(noisy_path)
        clean_audio = self._robust_load(clean_path)
        
        if noisy_audio is None or clean_audio is None:
            return torch.zeros(3, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE), torch.zeros(1, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)
            

        # STFT
        noisy_stft = librosa.stft(noisy_audio, n_fft=self.config.N_FFT, 
                                  hop_length=self.config.HOP_LENGTH, win_length=self.config.WIN_LENGTH)
        clean_stft = librosa.stft(clean_audio, n_fft=self.config.N_FFT, 
                                  hop_length=self.config.HOP_LENGTH, win_length=self.config.WIN_LENGTH)
        
        noisy_mag = np.abs(noisy_stft)
        clean_mag = np.abs(clean_stft)
        
        # Ensure both spectrograms have the same number of frames
        min_frames = min(noisy_mag.shape[1], clean_mag.shape[1])
        noisy_mag = noisy_mag[:, :min_frames]
        clean_mag = clean_mag[:, :min_frames]
        
        # Ideal Binary Mask (IBM): Nilai 1 jika magnitudo suara dominan
        mask = (clean_mag > 0.5 * noisy_mag).astype(np.float32)
        
        # Log-scale Normalize Input
        noisy_mag_db = librosa.amplitude_to_db(noisy_mag, ref=np.max)
        noisy_mag_norm = (noisy_mag_db - noisy_mag_db.min()) / (noisy_mag_db.max() - noisy_mag_db.min() + 1e-8)
        
        # Pad/Crop to target IMAGE_SIZE x IMAGE_SIZE
        noisy_mag_norm = self._pad_or_crop_spectrogram(noisy_mag_norm)
        mask = self._pad_or_crop_spectrogram(mask)
        
        # Add channel dimension
        x = torch.from_numpy(noisy_mag_norm).unsqueeze(0).float()
        y = torch.from_numpy(mask).unsqueeze(0).float()
        
        # Duplikasi ke 3 channel: (1,256,256) -> (3,256,256) sesuai paper 256x256x3
        x = x.repeat(3, 1, 1)
        
        return x, y

def get_dataloader(noisy_dir, clean_dir, config, shuffle=True):
    dataset = BirdAudioDataset(noisy_dir, clean_dir, config)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=shuffle, num_workers=0)
