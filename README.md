<div align="center">
  
# ViTVS: Deep Visual Audio Denoising

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/PyTorch_Lightning-2.0%2B-792ee5.svg)](https://www.pytorchlightning.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*An advanced Vision Transformer approach for Audio Source Separation and Denoising.*

</div>

---

## Overview

**ViTVS (Vision Transformer Segmentation)** transforms the complex problem of audio denoising into an image segmentation task. By converting raw audio waveforms into **Short-Time Fourier Transform (STFT)** spectrograms, we leverage the power of Vision Transformers (ViT) to predict an *Ideal Binary Mask (IBM)*. This mask effectively filters out background noise, leaving only the pristine target sound (e.g., Bird Vocalizations or Human Speech).

This repository contains a highly optimized, production-ready implementation of ViTVS using PyTorch Lightning, designed for scalability and reproducibility.

---

## Key Features

- **Transformers for Audio**: Utilizes self-attention mechanisms to capture long-range dependencies in audio spectrograms.
- **Robust Preprocessing**: Built-in Log-scale normalization and robust tensor padding/cropping for variable-length audio.
- **Lightning Fast**: Powered by PyTorch Lightning for effortless multi-GPU training, automatic checkpointing, and half-precision (FP16) support.
- **Evaluation Suite**: Built-in metrics computation including **F1-Score, IoU**, and **Signal-to-Distortion Ratio (SDR)**.

---

## Tech Stack

Our implementation is built on top of the following modern technologies:

| Category | Technology |
| :--- | :--- |
| **Core Framework** | [PyTorch](https://pytorch.org/) (Deep Learning) |
| **Training Engine** | [PyTorch Lightning](https://www.pytorchlightning.ai/) (Scalability & Boilerplate reduction) |
| **Audio Processing** | [Librosa](https://librosa.org/) & [TorchAudio](https://pytorch.org/audio/stable/index.html) (STFT/ISTFT operations) |
| **Tensor Ops** | [Einops](https://einops.rocks/) (Elegant tensor reshaping and rearrangement) |
| **Evaluation** | [TorchMetrics](https://torchmetrics.readthedocs.io/) (F1, Jaccard Index, SDR computation) |

---

## Project Structure

```text
bird-denoising-new/
├── src/
│   ├── config.py             # Global hyperparameters & system configurations
│   ├── dataset.py            # Custom PyTorch Dataset for Audio-to-Spectrogram & IBM masking
│   ├── model.py              # Core ViT Encoder & Decoder architecture
│   └── lightning_module.py   # PyTorch Lightning wrapper (Training/Val loops & Optimizers)
├── train.py                  # Entry point for training the model
├── test.py                   # Single-file inference script (Denoise a .wav file)
├── evaluate.py               # Mass-evaluation script for quantitative metrics (SDR, F1, IoU)
└── requirements.txt          # Python dependencies
```

---

## Getting Started

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/bird-denoising-new.git
cd bird-denoising-new
pip install -r requirements.txt
```

### 2. Dataset Preparation

Organize your paired audio dataset (Noisy vs Clean) into the following directory structure:

```text
data/
├── train/
│   ├── noisy/     # Noisy training audio files (.wav)
│   └── clean/     # Clean (Ground Truth) training audio files (.wav)
├── valid/
│   ├── noisy/
│   └── clean/
└── test/
    ├── noisy/
    └── clean/
```
*(Note: You can override these default paths directly in `src/config.py` or via CLI arguments).*

---

## Usage

### Training the Model

Start the training process. The script will automatically save the best models to the `checkpoints/` directory. PyTorch Lightning supports seamless resuming; if interrupted, simply re-run the command.

```bash
python train.py --epochs 100 --batch_size 8 --lr 0.0002
```

To monitor training curves (Loss) in real-time:
```bash
tensorboard --logdir lightning_logs/
```

### Evaluation (Quantitative)

Evaluate the trained model on your unseen Test dataset to compute the final **SDR, F1, and IoU** metrics:

```bash
python evaluate.py \
  --ckpt checkpoints/vitvs-best-val-epoch23-val_loss0.213.ckpt \
  --noisy_dir data/test/noisy \
  --clean_dir data/test/clean
```

### Inference (Denoise a single file)

Want to clean a real-world noisy audio file? Run the inference script:

```bash
python test.py \
  --ckpt checkpoints/vitvs-best-val-epoch23-val_loss0.213.ckpt \
  --input sample_noisy.wav \
  --output result_clean.wav
```

---

## Evaluation Results

*Example results after training on a subset of the dataset:*

| Metric | Score | Description |
| :--- | :--- | :--- |
| **F1 Score / Dice** | `88.30%` | Harmonic mean of precision and recall for mask prediction. |
| **IoU (Jaccard Index)** | `80.90%` | Area of Overlap / Area of Union between predicted and Ideal Binary Mask. |
| **SDR** | `11.04 dB` | Signal-to-Distortion Ratio (Higher is better clarity). |

---

## Acknowledgments

This implementation is inspired by the paper **"BirdSoundsDenoising: Deep Visual Audio Denoising for Bird Sounds" (WACV 2023)**. 

- **Original Paper:** [arXiv:2406.09167](https://arxiv.org/abs/2406.09167)
- **Official Dataset:** [Zenodo Record 7191406](https://zenodo.org/records/7191406)

---
<div align="center">
  <i>Built for better, cleaner audio representation.</i>
</div>
