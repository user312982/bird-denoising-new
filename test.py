import os
import argparse
import librosa
import numpy as np
import soundfile as sf
import torch

from src.config import Config
from src.lightning_module import ViTVSLightningModule

def test_denoising(checkpoint_path, audio_path, output_path='cleaned_audio.wav'):
    print(f"Loading model weights from {checkpoint_path}...")
    model = ViTVSLightningModule.load_from_checkpoint(checkpoint_path, config=Config)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    print(f"Processing audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=Config.SR)
    stft = librosa.stft(audio, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH, win_length=Config.WIN_LENGTH)
    mag = np.abs(stft)
    phase = np.angle(stft)
    
    # Normalize Magnitude
    mag_db = librosa.amplitude_to_db(mag, ref=np.max)
    mag_norm = (mag_db - mag_db.min()) / (mag_db.max() - mag_db.min() + 1e-8)
    
    target_size = Config.IMAGE_SIZE
    h, w = mag_norm.shape
    
    # Pad frequency axis if necessary
    if h < target_size:
        mag_norm = np.pad(mag_norm, ((0, target_size - h), (0, 0)), mode='constant')
        mag_ori_padded = np.pad(mag, ((0, target_size - h), (0, 0)), mode='constant')
        phase_padded = np.pad(phase, ((0, target_size - h), (0, 0)), mode='constant')
    else:
        mag_norm = mag_norm[:target_size, :]
        mag_ori_padded = mag[:target_size, :]
        phase_padded = phase[:target_size, :]

    reconstructed_mag = np.zeros_like(mag_ori_padded)
    
    # Process chunks over time window
    with torch.no_grad():
        for i in range(0, w, target_size):
            chunk = mag_norm[:, i:i+target_size]
            actual_chunk_w = chunk.shape[1]
            
            if actual_chunk_w < target_size:
                chunk = np.pad(chunk, ((0, 0), (0, target_size - actual_chunk_w)), mode='constant')
                
            x_tensor = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).float().to(device)
            
            # Forward pass to get Binary Mask
            predicted_mask = model(x_tensor).squeeze().cpu().numpy()
            valid_mask = predicted_mask[:, :actual_chunk_w]
            
            # Apply Mask (Dot Product)
            reconstructed_mag[:, i:i+actual_chunk_w] = mag_ori_padded[:, i:i+actual_chunk_w] * valid_mask
            
    # Pad back to original frequency bins if it was cropped
    if reconstructed_mag.shape[0] < h:
        pad_size = h - reconstructed_mag.shape[0]
        reconstructed_mag = np.pad(reconstructed_mag, ((0, pad_size), (0, 0)), mode='constant')
    else:
        reconstructed_mag = reconstructed_mag[:h, :]
        
    phase_original = phase[:h, :]
    # ISTFT to reconstruct waveform
    complex_spec = reconstructed_mag * np.exp(1j * phase_original)
    clean_audio = librosa.istft(complex_spec, hop_length=Config.HOP_LENGTH, win_length=Config.WIN_LENGTH)
    
    sf.write(output_path, clean_audio, sr)
    print(f"Success! Denoised audio saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test ViTVS Audio Denoising')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint (.ckpt file)')
    parser.add_argument('--input', type=str, required=True, help='Path to noisy input .wav file')
    parser.add_argument('--output', type=str, default='cleaned_audio.wav', help='Path to save clean .wav file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.ckpt):
        print(f"Error: Checkpoint file not found: {args.ckpt}")
    elif not os.path.exists(args.input):
        print(f"Error: Input audio file not found: {args.input}")
    else:
        test_denoising(args.ckpt, args.input, args.output)
