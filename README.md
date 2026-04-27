# ViTVS Audio Denoising

This repository contains the PyTorch Lightning implementation of Vision Transformer Segmentation (ViTVS) specifically designed for Audio Denoising (e.g., separating bird sounds from noise). 

It converts audio into spectrograms, passes them through a ViT Encoder-Decoder architecture to generate a Binary Mask, and reconstructs the clean audio using ISTFT.

## Project Structure

- `config.py`: Hyperparameters and dataset paths.
- `dataset.py`: PyTorch `Dataset` for converting `.wav` to spectrograms and computing the Ideal Binary Mask.
- `model.py`: PyTorch implementation of the ViTVS Encoder and Decoder.
- `lightning_module.py`: PyTorch Lightning wrapper for training and optimization.
- `train.py`: CLI script to train the model with automatic resume capability.
- `test.py`: CLI script to denoise a single audio file using a trained checkpoint.

## Setup

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare your dataset:
   Place your audio files in the `data/` folder (or specify custom paths in `config.py`):
   - `data/train/noisy/*.wav` and `data/train/clean/*.wav`
   - `data/valid/noisy/*.wav` and `data/valid/clean/*.wav`
   - `data/test/noisy/*.wav` and `data/test/clean/*.wav`

## Training

Run the training script. It will automatically save checkpoints to the `checkpoints/` folder. If training is interrupted, simply run the command again and it will resume from the last saved `.ckpt`.

```bash
python train.py
```

You can also override hyperparameters via CLI:
```bash
python train.py --epochs 100 --batch_size 8 --lr 0.0001
```

## Inference / Testing

To denoise an audio file using a trained checkpoint:

```bash
python test.py --ckpt checkpoints/vitvs_denoising/last.ckpt --input test_noisy.wav --output test_clean.wav
```
