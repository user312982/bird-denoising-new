import os

class Config:
    # Training Parameters
    EPOCHS = 100
    BATCH_SIZE = 8
    LR = 5e-5
    WEIGHT_DECAY = 5e-4
    
    # Audio parameters
    SR = 16000
    N_FFT = 512
    HOP_LENGTH = 256
    WIN_LENGTH = 512
    
    # Model parameters (ViT)
    IMAGE_SIZE = 256  # Size of Spectrogram patch (256x256)
    PATCH_SIZE = 16   # Patch size for ViT
    DIM = 512
    DEPTH = 12        # 12 Layers of ViT Blocks
    HEADS = 8
    MLP_DIM = 2048
    IN_CHANNELS = 3   # 3-channel input (256x256x3) sesuai paper
    
    # Default Paths (Can be overridden via CLI)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_PATH = os.path.join(BASE_DIR, 'data')
    NOISY_TRAIN_DIR = os.path.join(DATASET_PATH, 'train', 'noisy')
    CLEAN_TRAIN_DIR = os.path.join(DATASET_PATH, 'train', 'clean')
    NOISY_VALID_DIR = os.path.join(DATASET_PATH, 'valid', 'noisy')
    CLEAN_VALID_DIR = os.path.join(DATASET_PATH, 'valid', 'clean')
    NOISY_TEST_DIR = os.path.join(DATASET_PATH, 'test', 'noisy')
    CLEAN_TEST_DIR = os.path.join(DATASET_PATH, 'test', 'clean')
    CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints', 'vitvs_denoising')

# Ensure checkpoint dir exists
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
