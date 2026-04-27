import os
import glob
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

from src.config import Config
from src.dataset import get_dataloader
from src.lightning_module import ViTVSLightningModule

def get_latest_checkpoint(checkpoint_dir):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getctime)

def main(args):
    # Override config with CLI args
    Config.EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.LR = args.lr
    
    if args.noisy_dir: Config.NOISY_TRAIN_DIR = args.noisy_dir
    if args.clean_dir: Config.CLEAN_TRAIN_DIR = args.clean_dir
    if args.checkpoint_dir: Config.CHECKPOINT_DIR = args.checkpoint_dir

    print(f"Preparing dataset from: {Config.NOISY_TRAIN_DIR} and {Config.CLEAN_TRAIN_DIR}")
    train_loader = get_dataloader(Config.NOISY_TRAIN_DIR, Config.CLEAN_TRAIN_DIR, Config)
    
    if len(train_loader.dataset) == 0:
        print("Error: Dataset is empty. Please check your data paths.")
        return

    model = ViTVSLightningModule(Config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=Config.CHECKPOINT_DIR,
        filename='vitvs-epoch{epoch:02d}',
        auto_insert_metric_name=False,
        save_top_k=2,
        monitor='train_loss',
        save_last=True
    )

    trainer = pl.Trainer(
        max_epochs=Config.EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback]
    )

    latest_ckpt = get_latest_checkpoint(Config.CHECKPOINT_DIR)

    if latest_ckpt:
        print(f"\n[INFO] Resuming training from Checkpoint: {latest_ckpt}")
        trainer.fit(model, train_dataloaders=train_loader, ckpt_path=latest_ckpt)
    else:
        print("\n[INFO] Starting training from scratch (Epoch 0)")
        trainer.fit(model, train_dataloaders=train_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ViTVS Audio Denoising Model')
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=Config.LR, help='Learning rate')
    parser.add_argument('--noisy_dir', type=str, default='', help='Path to noisy train data')
    parser.add_argument('--clean_dir', type=str, default='', help='Path to clean train data')
    parser.add_argument('--checkpoint_dir', type=str, default='', help='Directory to save checkpoints')
    
    args = parser.parse_args()
    main(args)
