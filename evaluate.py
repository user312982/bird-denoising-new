import os
import glob
import argparse
import librosa
import numpy as np
import torch
from tqdm import tqdm
from src.config import Config
from src.lightning_module import ViTVSLightningModule

# Pastikan Anda sudah menjalankan: pip install torchmetrics
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex
from torchmetrics.audio import SignalDistortionRatio

def evaluate_model(checkpoint_path, test_noisy_dir, test_clean_dir):
    print(f"Loading model dari {checkpoint_path}...")
    model = ViTVSLightningModule.load_from_checkpoint(checkpoint_path, config=Config)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    noisy_files = sorted(glob.glob(os.path.join(test_noisy_dir, '*.wav')))
    clean_files = sorted(glob.glob(os.path.join(test_clean_dir, '*.wav')))
    
    if len(noisy_files) == 0 or len(clean_files) == 0:
        print("Error: Tidak ada file audio ditemukan di direktori test!")
        return
        
    assert len(noisy_files) == len(clean_files), "Jumlah file noisy dan clean harus sama!"

    # Inisialisasi Metrics
    f1_metric = BinaryF1Score().to(device)
    iou_metric = BinaryJaccardIndex().to(device)
    sdr_metric = SignalDistortionRatio().to(device)

    all_f1 = []
    all_iou = []
    all_sdr = []

    print(f"Mengevaluasi {len(noisy_files)} file...")
    
    with torch.no_grad():
        for noisy_path, clean_path in tqdm(zip(noisy_files, clean_files), total=len(noisy_files)):
            # 1. Load Audio
            noisy_audio, sr = librosa.load(noisy_path, sr=Config.SR)
            clean_audio, _ = librosa.load(clean_path, sr=Config.SR)
            
            # Pastikan panjangnya sama
            min_len = min(len(noisy_audio), len(clean_audio))
            noisy_audio = noisy_audio[:min_len]
            clean_audio = clean_audio[:min_len]

            # 2. STFT
            noisy_stft = librosa.stft(noisy_audio, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH, win_length=Config.WIN_LENGTH)
            clean_stft = librosa.stft(clean_audio, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH, win_length=Config.WIN_LENGTH)
            
            noisy_mag = np.abs(noisy_stft)
            clean_mag = np.abs(clean_stft)
            noisy_phase = np.angle(noisy_stft)
            
            # 3. Buat Ground Truth IBM (Ideal Binary Mask)
            # Nilai 1 jika clean_mag dominan, 0 jika noise dominan
            true_mask = (clean_mag > 0.5 * noisy_mag).astype(np.float32)

            # Log-scale Normalize Noisy Input
            noisy_mag_db = librosa.amplitude_to_db(noisy_mag, ref=np.max)
            noisy_mag_norm = (noisy_mag_db - noisy_mag_db.min()) / (noisy_mag_db.max() - noisy_mag_db.min() + 1e-8)
            
            target_size = Config.IMAGE_SIZE
            h, w = noisy_mag_norm.shape
            
            # Pad frequency axis jika kurang dari IMAGE_SIZE
            if h < target_size:
                noisy_mag_norm = np.pad(noisy_mag_norm, ((0, target_size - h), (0, 0)), mode='constant')
                noisy_mag = np.pad(noisy_mag, ((0, target_size - h), (0, 0)), mode='constant')
                true_mask_padded = np.pad(true_mask, ((0, target_size - h), (0, 0)), mode='constant')
            else:
                noisy_mag_norm = noisy_mag_norm[:target_size, :]
                noisy_mag = noisy_mag[:target_size, :]
                true_mask_padded = true_mask[:target_size, :]

            reconstructed_mag = np.zeros_like(noisy_mag)
            full_predicted_mask = np.zeros_like(true_mask_padded)
            
            # 4. Prediksi per chunk (sliding window)
            for i in range(0, w, target_size):
                chunk = noisy_mag_norm[:, i:i+target_size]
                actual_chunk_w = chunk.shape[1]
                
                if actual_chunk_w < target_size:
                    chunk = np.pad(chunk, ((0, 0), (0, target_size - actual_chunk_w)), mode='constant')
                    
                x_tensor = torch.from_numpy(chunk).unsqueeze(0).float()
                x_tensor = x_tensor.repeat(1, 3, 1, 1).to(device)
                
                logits = model(x_tensor)
                predicted_chunk_mask = torch.sigmoid(logits).squeeze().cpu().numpy()
                valid_mask = predicted_chunk_mask[:, :actual_chunk_w]
                
                full_predicted_mask[:, i:i+actual_chunk_w] = valid_mask
                reconstructed_mag[:, i:i+actual_chunk_w] = noisy_mag[:, i:i+actual_chunk_w] * valid_mask
                
            # 5. Hitung Metrik Segmentasi (F1, IoU)
            # Bandingkan full_predicted_mask dengan true_mask_padded
            tensor_pred_mask = torch.from_numpy(full_predicted_mask).to(device)
            tensor_true_mask = torch.from_numpy(true_mask_padded).int().to(device)
            
            f1_val = f1_metric(tensor_pred_mask, tensor_true_mask)
            iou_val = iou_metric(tensor_pred_mask, tensor_true_mask)
            
            all_f1.append(f1_val.item())
            all_iou.append(iou_val.item())
            
            # 6. Rekonstruksi Audio (ISTFT) untuk SDR
            if reconstructed_mag.shape[0] < true_mask.shape[0]:
                pad_size = true_mask.shape[0] - reconstructed_mag.shape[0]
                reconstructed_mag = np.pad(reconstructed_mag, ((0, pad_size), (0, 0)), mode='constant')
            else:
                reconstructed_mag = reconstructed_mag[:true_mask.shape[0], :]
                
            phase_original = noisy_phase[:true_mask.shape[0], :]
            complex_spec = reconstructed_mag * np.exp(1j * phase_original)
            denoised_audio = librosa.istft(complex_spec, hop_length=Config.HOP_LENGTH, win_length=Config.WIN_LENGTH)
            
            # Potong denoised audio agar panjangnya sama dengan clean audio
            min_audio_len = min(len(denoised_audio), len(clean_audio))
            denoised_audio = denoised_audio[:min_audio_len]
            clean_audio_target = clean_audio[:min_audio_len]
            
            # 7. Hitung SDR (Bandingkan denoised_audio dengan clean_audio_target)
            tensor_denoised = torch.from_numpy(denoised_audio).to(device)
            tensor_clean = torch.from_numpy(clean_audio_target).to(device)
            
            sdr_val = sdr_metric(tensor_denoised, tensor_clean)
            all_sdr.append(sdr_val.item())

    # Cetak Hasil Akhir (Rata-rata)
    print("\n" + "="*40)
    print("HASIL EVALUASI MODEL")
    print("="*40)
    print(f"Rata-rata F1 Score / Dice: {np.mean(all_f1) * 100:.2f}")
    print(f"Rata-rata IoU            : {np.mean(all_iou) * 100:.2f}")
    print(f"Rata-rata SDR            : {np.mean(all_sdr):.2f} dB")
    print("="*40)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluasi ViTVS dengan Test Dataset')
    parser.add_argument('--ckpt', type=str, required=True, help='Path ke file model (.ckpt)')
    parser.add_argument('--noisy_dir', type=str, required=True, help='Path ke folder test noisy audio')
    parser.add_argument('--clean_dir', type=str, required=True, help='Path ke folder test clean audio')
    
    args = parser.parse_args()
    evaluate_model(args.ckpt, args.noisy_dir, args.clean_dir)
