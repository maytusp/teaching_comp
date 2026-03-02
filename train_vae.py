import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

import numpy as np

from disent.metrics import metric_dci

from utils.eval_recon import ImageLogger, run_final_eval
from utils.load_data import make_challenging_dataloaders as make_dataloaders
from utils.linear_probe import evaluate_linear_probe

from models import LitBetaVae


if __name__ == "__main__":
    BETA_VAL = 1.0
    MAX_EPOCHS = 50
    EXPERIMENT_NAME = f"beta_vae_b{BETA_VAL}"
    print(f"EXPERIMENT_NAME: {EXPERIMENT_NAME}")
    # Define output paths based on experiment name
    CHECKPOINT_DIR = os.path.join("checkpoints", EXPERIMENT_NAME)
    VIS_DIR = f"results/{EXPERIMENT_NAME}/visualizations"
    EVAL_DIR = f"results/{EXPERIMENT_NAME}/evaluation"
    # ---------------------

    # 1. Data
    train_dl, val_dl, test_dl, full_ds = make_dataloaders()
    
    # 2. Model
    model = LitBetaVae(z_size=64, beta=BETA_VAL)
    
    # 3. Trainer with Checkpointing
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss", 
        dirpath=CHECKPOINT_DIR,  # Saves to checkpoints/beta_vae_b4.0_v1/
        filename="best_vae"
    )
    
    image_logger = ImageLogger(save_dir=VIS_DIR)

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_callback, image_logger]
    )
    
    trainer.fit(model, train_dl, val_dl)
    run_final_eval(model, test_dl, full_ds, output_dir=EVAL_DIR)
    evaluate_linear_probe(model, train_dl, val_dl, test_dl, model.device)