# Rencana Implementasi: Mengadaptasi ViTVS Berdasarkan Paper Asli (arXiv:2406.09167)

Berdasarkan analisis paper asli **"Vision Transformer Segmentation for Visual Bird Sound Denoising" (ViTVS)** karya Sahil Kumar dkk. (2024), project ini sudah berada di jalur yang tepat, namun ada beberapa detail arsitektur dan hyperparameter yang perlu disesuaikan agar *plek-ketiplek* (identik) dengan spesifikasi di paper.

## Mengapa Pendekatan Paper Ini Sangat Bagus?
1. **Representasi Jarak Jauh (Long-Range Dependencies):** Berbeda dengan CNN/U-Net biasa yang hanya melihat fitur lokal (sekitarnya), arsitektur ViTVS dengan mekanisme *Self-Attention* global (12 layer di Encoder & 12 layer di Decoder) mampu menangkap pola suara burung secara keseluruhan (seperti nada harmonik yang tersebar jauh di sumbu frekuensi dan waktu).
2. **Kapasitas Multi-Scale:** Model ini memperlakukan spectrogram audio murni sebagai gambar, membaginya ke dalam *patches*, dan mempelajari *semantic segmentation* untuk memisahkan burung vs noise.
3. **Regularisasi yang Kuat:** Penggunaan **AdamW** dengan *weight decay* mencegah model transformer (yang memiliki ~87M parameter) dari *overfitting*, sesuatu yang sering gagal dilakukan oleh optimizer Adam biasa.
4. **Normalisasi Khusus (Batch Normalization):** Paper secara spesifik menyebutkan penggunaan *Batch Normalization (BN)* pada gambar spectrogram sebelum diubah menjadi *patches*, yang menstabilkan input ke dalam transformer.

---

## User Review Required

> [!IMPORTANT]
> Mohon direview apakah Anda setuju dengan perubahan **channel gambar**. Paper menyebutkan me-resize gambar menjadi `256 x 256 x 3` (3 channel seperti RGB). Hal ini sering dilakukan agar ViT bisa memproses fitur lebih kaya atau menggunakan *pre-trained weights*. Apakah Anda setuju kita mengubah input spectrogram menjadi 3 channel (dengan menduplikasi nilai spectrogram ke 3 channel)?

> [!WARNING]
> Memori GPU: Paper menggunakan *Batch Size* = 8. Jika Anda melatih model di Colab GPU T4 (16GB VRAM), *Batch Size* 8 untuk ViT dengan kedalaman 12 layer mungkin akan menyebabkan *Out of Memory* (OOM). Apakah kita akan tetap mencoba `BATCH_SIZE = 8`, atau membatasinya sesuai kapasitas (misal 4)?

---

## Proposed Changes

### Konfigurasi Hyperparameter (src/config.py)
Menyesuaikan hyperparameter persis seperti yang dijelaskan di "Section 4.1. Implementation details" pada paper.
#### [MODIFY] config.py
- Ubah `EPOCHS` dari 50 menjadi **100**.
- Ubah `BATCH_SIZE` dari 4 menjadi **8**.
- Ubah `LR` (Learning Rate) dari 1e-4 menjadi **5e-5**.
- Tambahkan hyperparameter `WEIGHT_DECAY = 5e-4`.
- Tambahkan `IN_CHANNELS = 3` (sesuai spesifikasi `256 x 256 x 3` di paper).

---

### Arsitektur Model (src/model.py)
Menyesuaikan transformasi input agar menyertakan *Batch Normalization* seperti di "Section 3.1. Input Transformation".
#### [MODIFY] model.py
- **ViTVS_Encoder**: Tambahkan layer `nn.BatchNorm2d(channels)` persis sebelum proses `Rearrange` (Image to Patch embedding). Sesuai dengan rumus di paper: `X = Linear(ITP(BN(I)))`.
- **ViTVS.__init__**: Update inisialisasi class agar meneruskan jumlah channel dari config ke encoder dan decoder (contoh: `channels=config.IN_CHANNELS`), karena saat ini masih di-hardcode dengan `channels=1`.
- Hapus `torch.sigmoid()` di fungsi `forward` utama model. Kita akan memindahkannya ke *Loss Function* menggunakan `BCEWithLogitsLoss` demi stabilitas numerik yang lebih baik (setara dengan *Negative Log-Likelihood* pada klasifikasi biner yang disebut di paper).

---

### PyTorch Lightning Module & Optimizer (src/lightning_module.py)
Mengganti optimizer dari Adam biasa ke AdamW dengan weight decay.
#### [MODIFY] lightning_module.py
- **configure_optimizers**: Ganti `torch.optim.Adam` menjadi `torch.optim.AdamW(self.parameters(), lr=self.config.LR, weight_decay=self.config.WEIGHT_DECAY)`.
- Ganti `nn.BCELoss()` menjadi `nn.BCEWithLogitsLoss()` untuk menangani *raw logits* langsung dari model, memberikan stabilitas *gradient* yang lebih baik layaknya *log-softmax* di paper.

---

### Pipeline Data (src/dataset.py)
Menyesuaikan bentuk tensor input agar memiliki 3 dimensi channel.
#### [MODIFY] dataset.py
- Duplikasi tensor spectrogram dari 1 channel menjadi 3 channel: `x = x.repeat(3, 1, 1)` agar *shape* input menjadi `(3, 256, 256)` sesuai parameter `256 x 256 x 3` di paper.

---

### Inference Script (test.py)
Menambahkan aktivasi sigmoid yang dihapus dari arsitektur model saat melakukan inferensi.
#### [MODIFY] test.py
- Di dalam iterasi inferensi (waktu mengekstrak Binary Mask dari model), tambahkan pemanggilan `torch.sigmoid()` secara manual ke hasil *forward pass* model. Karena model sekarang mengembalikan *raw logits*, memanggil sigmoid memastikan mask tetap berada di rentang nilai probabilistik [0, 1].

---

## Verification Plan
1. Menjalankan `python train.py` secara lokal untuk memastikan model dapat melakukan proses *forward* dan *backward pass* tanpa *Error/OOM*.
2. Memeriksa kembali arsitektur model menggunakan `print(model)` untuk memastikan `BatchNorm2d` sudah berada di posisi yang benar sebelum `Rearrange` dan `IN_CHANNELS` diterima dengan baik.
3. Menjalankan `python test.py` untuk memastikan `sigmoid` berhasil mengubah logits menjadi map binary [0, 1] dan file audio sukses dihasilkan.
4. Memastikan *loss* mulai menurun menggunakan `AdamW` dengan learning rate baru.
