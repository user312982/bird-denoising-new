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
    if args.val_noisy_dir: Config.NOISY_VALID_DIR = args.val_noisy_dir
    if args.val_clean_dir: Config.CLEAN_VALID_DIR = args.val_clean_dir
    if args.checkpoint_dir: Config.CHECKPOINT_DIR = args.checkpoint_dir

    print(f"Preparing train dataset from: {Config.NOISY_TRAIN_DIR} and {Config.CLEAN_TRAIN_DIR}")
    train_loader = get_dataloader(Config.NOISY_TRAIN_DIR, Config.CLEAN_TRAIN_DIR, Config)
    
    print(f"Preparing validation dataset from: {Config.NOISY_VALID_DIR} and {Config.CLEAN_VALID_DIR}")
    val_loader = get_dataloader(Config.NOISY_VALID_DIR, Config.CLEAN_VALID_DIR, Config, shuffle=False)
    
    if len(train_loader.dataset) == 0:
        print("Error: Dataset is empty. Please check your data paths.")
        return

    model = ViTVSLightningModule(Config)

    val_checkpoint = ModelCheckpoint(
        dirpath=Config.CHECKPOINT_DIR,
        filename='vitvs-best-val-epoch{epoch:02d}-val_loss{val_loss:.3f}',
        auto_insert_metric_name=False,
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        save_last=True
    )

    train_checkpoint = ModelCheckpoint(
        dirpath=Config.CHECKPOINT_DIR,
        filename='vitvs-best-train-epoch{epoch:02d}-train_loss{train_loss:.3f}',
        auto_insert_metric_name=False,
        save_top_k=1,
        monitor='train_loss',
        mode='min',
    )

    trainer = pl.Trainer(
        max_epochs=Config.EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[val_checkpoint, train_checkpoint]
    )

    latest_ckpt = get_latest_checkpoint(Config.CHECKPOINT_DIR)

    if latest_ckpt:
        print(f"\n[INFO] Resuming training from Checkpoint: {latest_ckpt}")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=latest_ckpt)
    else:
        print("\n[INFO] Starting training from scratch (Epoch 0)")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ViTVS Audio Denoising Model')
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=Config.LR, help='Learning rate')
    parser.add_argument('--noisy_dir', type=str, default='', help='Path to noisy train data')
    parser.add_argument('--clean_dir', type=str, default='', help='Path to clean train data')
    parser.add_argument('--val_noisy_dir', type=str, default='', help='Path to noisy validation data')
    parser.add_argument('--val_clean_dir', type=str, default='', help='Path to clean validation data')
    parser.add_argument('--checkpoint_dir', type=str, default='', help='Directory to save checkpoints')
    
    args = parser.parse_args()
    main(args)
