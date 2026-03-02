import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. DEFINE NETWORKS LOCALLY
# ==========================================

class Encoder(nn.Module):
    def __init__(self, z_size, input_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(),
        )
        self.flatten_size = 64 * 4 * 4
        self.fc_mu = nn.Linear(self.flatten_size, z_size)
        self.fc_logvar = nn.Linear(self.flatten_size, z_size)

    def forward(self, x):
        h = self.net(x)
        # FIX: Use .reshape() instead of .view() to handle non-contiguous memory
        h = h.reshape(h.size(0), -1) 
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, z_size, output_channels=3):
        super().__init__()
        self.fc = nn.Linear(z_size, 64 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, 4, 2, 1) 
        )

    def forward(self, z):
        h = self.fc(z)
        # FIX: Use .reshape() here too
        h = h.reshape(h.size(0), 64, 4, 4)
        x_recon = self.net(h)
        return x_recon

# ==========================================
# 2. LIGHTNING MODULE
# ==========================================

class LitBetaVae(pl.LightningModule):
    def __init__(self, z_size=10, beta=1.0, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(z_size)
        self.decoder = Decoder(z_size)
        self.beta = beta
        self.lr = lr

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def training_step(self, batch, batch_idx):
        # 1. Extract Data
        if isinstance(batch, dict):
            x = batch['x_targ']
        elif isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
            
        if isinstance(x, (list, tuple)):
            x = x[0]

        # 2. Preprocessing
        # Permute (H, W, C) -> (C, H, W) if needed
        if x.ndim == 4 and x.shape[-1] == 3: 
            x = x.permute(0, 3, 1, 2)
            
        # Normalize uint8 -> float
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        # 3. Forward Pass
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)

        # 4. Loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.shape[0]
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
        loss = recon_loss + (self.beta * kl_loss)

        self.log("train_loss", loss, prog_bar=True)
        self.log("recon_loss", recon_loss)
        self.log("kl_loss", kl_loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
