import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision

class Encoder(nn.Module):
    def __init__(self, z_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(), # 32x32
            nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(), # 16x16
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(), # 8x8
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(), # 4x4
        )
        self.fc_mu = nn.Linear(64 * 4 * 4, z_size)
        self.fc_logvar = nn.Linear(64 * 4 * 4, z_size)

    def forward(self, x):
        h = self.net(x).reshape(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self, z_size):
        super().__init__()
        self.fc = nn.Linear(z_size, 64 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid() 
        )

    def forward(self, z):
        h = self.fc(z).reshape(z.size(0), 64, 4, 4)
        return self.net(h)

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

    def _get_x(self, batch):
        x = batch['x_targ'] if isinstance(batch, dict) else batch[0]
        if isinstance(x, (list, tuple)): x = x[0]
        # Shape fix: (B, 64, 64, 3) -> (B, 3, 64, 64)
        if x.shape[-1] == 3: x = x.permute(0, 3, 1, 2)
        if x.dtype == torch.uint8: x = x.float() / 255.0
        return x

    def training_step(self, batch, batch_idx):
        x = self._get_x(batch)
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.shape[0]
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
        loss = recon_loss + (self.beta * kl_loss)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self._get_x(batch)
        mu, logvar = self.encoder(x)
        x_recon = self.decoder(mu)
        loss = F.mse_loss(x_recon, x, reduction='sum') / x.shape[0]
        self.log("val_loss", loss, prog_bar=True)
        
        # Save a sample for visualization
        if batch_idx == 0:
            self.sample_imgs = (x[:8], x_recon[:8])
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class LitTeachingVae(pl.LightningModule):
    def __init__(self, z_size=10, beta=4.0, lr=1e-4, config=None):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Teacher Components (The main VAE - sees clean data)
        self.teacher_encoder = Encoder(z_size)
        self.decoder = Decoder(z_size)
        
        # Student Components (List of Encoders - see noisy data)
        self.students = nn.ModuleList([
            Encoder(z_size) for _ in range(config['n_students'])
        ])
        
        self.beta = beta
        self.lr = lr

    def kl_divergence_gaussians(self, mu_q, logvar_q, mu_p, logvar_p):
        """
        Calculates KL(Q || P) for two diagonal multivariate Gaussians.
        
        Q = Student (The approximate/noisy distribution)
        P = Teacher (The true/clean distribution)
        
        Minimizing this w.r.t P (Teacher) forces the Teacher to be 
        "broad" enough to cover the Student's uncertainty (Robustness).
        """
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        
        # Standard analytical formula for KL divergence
        kl = 0.5 * torch.sum(
            logvar_p - logvar_q - 1.0 + 
            (var_q + (mu_q - mu_p).pow(2)) / var_p,
            dim=-1
        )
        return kl.mean()

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def _get_x(self, batch):
        x = batch['x_targ'] if isinstance(batch, dict) else batch[0]
        if isinstance(x, (list, tuple)): x = x[0]
        if x.shape[-1] == 3: x = x.permute(0, 3, 1, 2)
        if x.dtype == torch.uint8: x = x.float() / 255.0
        return x

    def training_step(self, batch, batch_idx):
        # 1. Clean Data (For Teacher)
        x_clean = self._get_x(batch)
        
        # 2. Noisy Data (For Students) - Forces robustness
        # The students see a corrupted version of the world
        noise = torch.randn_like(x_clean) * self.config['student_noise_std']
        x_noisy = x_clean + noise

        # ---------------------------
        # A. TEACHER FORWARD (VAE)
        # ---------------------------
        # Teacher outputs parameters for distribution P(z|x)
        mu_t, logvar_t = self.teacher_encoder(x_clean)
        
        # Standard VAE Flow
        z = self.reparameterize(mu_t, logvar_t)
        x_recon = self.decoder(z)
        
        # VAE Losses (Reconstruction + KL Prior)
        recon_loss = F.mse_loss(x_recon, x_clean, reduction='sum') / x_clean.shape[0]
        kl_prior = -0.5 * torch.sum(1 + logvar_t - mu_t.pow(2) - logvar_t.exp()) / x_clean.shape[0]
        vae_loss = recon_loss + (self.beta * kl_prior)

        # ---------------------------
        # B. STUDENT FORWARD (Teaching Regularization)
        # ---------------------------
        teaching_loss_total = 0.0
        
        for i, student in enumerate(self.students):
            # Heterogeneous Delays: Only update specific students at specific steps
            freq = self.config['update_freqs'][i]
            
            if self.global_step % freq == 0:
                # Student outputs parameters for distribution Q(z|x_noisy)
                mu_s, logvar_s = student(x_noisy)
                
                # UNIDIRECTIONAL KL (Student || Teacher)
                # Q = Student, P = Teacher
                # CRITICAL: We do NOT detach/stop_grad the Teacher!
                # We want the gradient to flow into the Teacher so it adapts its P 
                # to better accommodate the Student's Q.
                loss_kl = self.kl_divergence_gaussians(
                    mu_s, logvar_s,   # Q (The Student)
                    mu_t, logvar_t    # P (The Teacher - Gradient flows here!)
                )
                
                teaching_loss_total += loss_kl

        # ---------------------------
        # C. TOTAL LOSS
        # ---------------------------
        # Combine standard VAE objective with the teaching regularizer
        total_loss = vae_loss + (self.config['teaching_lambda'] * teaching_loss_total)
        
        # Logging
        self.log("train_loss", total_loss, prog_bar=True)
        self.log("vae_loss", vae_loss)
        self.log("teach_loss", teaching_loss_total)
        
        return total_loss

    def on_train_epoch_start(self):
        # Cyclic Re-initialization
        # Periodically resets students to prevent them from converging too early
        # and to force the teacher to re-explain concepts to a "fresh" mind.
        if self.current_epoch in self.config['reinit_epochs']:
            print(f"\n[Teaching Regularization] Re-initializing all students at Epoch {self.current_epoch}!")
            for student in self.students:
                student.apply(self.weight_reset)
    
    @staticmethod
    def weight_reset(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

    def validation_step(self, batch, batch_idx):
        x = self._get_x(batch)
        mu, _ = self.teacher_encoder(x)
        x_recon = self.decoder(mu)
        loss = F.mse_loss(x_recon, x, reduction='sum') / x.shape[0]
        self.log("val_loss", loss, prog_bar=True)
        # Save sample images for TensorBoard/Logging
        if batch_idx == 0: self.sample_imgs = (x[:8], x_recon[:8])
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # For evaluation, we expose the teacher's encoder as the primary 'encoder'
    @property
    def encoder(self):
        return self.teacher_encoder