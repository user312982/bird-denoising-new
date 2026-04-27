import pytorch_lightning as pl
import torch
import torch.nn as nn
from src.model import ViTVS

class ViTVSLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # Ensure config properties are serializable, avoiding the raw class object
        self.save_hyperparameters(
            {k: v for k, v in config.__dict__.items() if not k.startswith('__')}
        )
        self.config = config
        self.model = ViTVS(config)
        self.criterion = nn.BCELoss() # Binary Cross Entropy for mask prediction

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.LR)
        return optimizer
