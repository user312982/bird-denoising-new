# Product Requirements Document (PRD): ViTVS Code Reproduction (Notebook)

## 1. Tujuan Utama
Menulis kode dari nol untuk mereproduksi arsitektur **Vision Transformer Segmentation (ViTVS)** khusus untuk *Audio Denoising* (memisahkan suara burung dari noise). Hasil akhir harus berupa satu file `notebook.ipynb` yang siap dijalankan di Google Colab (GPU T4) tanpa error.

## 2. Referensi GitHub & Standar Pustaka (Wajib Dipakai)
Kode harus dirakit dengan mengadopsi pola dari repositori berikut:
1. **`lucidrains/vit-pytorch`**: Gunakan pola kode dari repositori ini untuk membangun blok **Encoder** (Vision Transformer dengan Multi-head Self-Attention).
2. **`qubvel/segmentation_models.pytorch`**: Gunakan pola dari repositori ini untuk mengelola bentuk **Decoder** dan proyeksi *output* menjadi Peta Segmentasi Biner (*Binary Mask*).
3. **`Lightning-AI/pytorch-lightning`**: Gunakan *framework* ini untuk membungkus model (sebagai `LightningModule`). Ini wajib untuk menyederhanakan *training loop* dan otomatisasi *checkpoint*.
4. **`librosa` / `torchaudio`**: Gunakan untuk konversi STFT (Audio ke Spectrogram) dan ISTFT (Spectrogram + Phase ke Audio).

## 3. Spesifikasi Arsitektur Pipeline
- **Input:** File `.wav` -> STFT -> Magnitude Spectrogram -> Normalisasi -> Potong jadi *patches* $p \times p$. (Pastikan *Phase* disimpan terpisah).
- **Model Utama:** - Encoder: 12 lapis ViT blocks.
  - Decoder: 12 lapis blok dengan *unfolding* untuk rekonstruksi gambar.
  - Output: Binary Mask (ukuran sama dengan input spectrogram).
- **Output:** Spectrogram input dikalikan (dot product) dengan Binary Mask -> ISTFT dengan *Phase* asli -> File `.wav` bersih.
- **Loss & Optimizer:** Binary Cross-Entropy (BCE) Loss dan Adam Optimizer.

## 4. Struktur File `notebook.ipynb` (Fokus Pembuatan Kode)
AI harus menghasilkan kode terstruktur untuk cell-cell berikut:

- **Cell 1: Setup & Import**
  - Instalasi Pytorch Lightning, librosa, torchaudio.
  - Automount Google Drive (`/content/drive`).
- **Cell 2: Konfigurasi & Hyperparameters**
  - `EPOCHS = 50`, `BATCH_SIZE = 4`, `LR = 1e-4`.
  - Definisi path dataset dan folder `/checkpoints/` di Google Drive.
- **Cell 3: Custom Dataset (PyTorch)**
  - Class DataLoader yang mengkonversi audio menjadi spectrogram (input) dan memuat mask murni (target).
- **Cell 4: Definisi Arsitektur ViTVS**
  - Implementasi class Pytorch murni untuk `ViTVS_Encoder` dan `ViTVS_Decoder`.
- **Cell 5: PyTorch Lightning Module**
  - Bungkus arsitektur ke dalam `pl.LightningModule`. Tulis fungsi `forward`, `training_step`, dan `configure_optimizers`.
- **Cell 6: Training Orchestration (Anti-Timeout Colab)**
  - Konfigurasi `ModelCheckpoint` untuk menyimpan file `.ckpt` ke GDrive.
  - Logika krusial: Jika file checkpoint sebelumnya ada di GDrive, `Trainer.fit()` harus otomatis melakukan *resume training*.
- **Cell 7: Modul Testing (Epoch 8 Evaluator)**
  - Fungsi `test_denoising(checkpoint_path, audio_path)`.
  - Fungsi untuk memuat bobot dari file `.ckpt` (misal dari Epoch 8), memproses satu file audio, dan menghasilkan file `.wav` hasil *denoising*.