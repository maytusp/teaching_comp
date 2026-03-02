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

from models import LitTeachingVae

TEACHING_CONFIG = {
    # Number of students
    "n_students": 2,

    "beta": 2.0,

    # input noise for student
    "student_noise_std": 0.1,
    
    # Heterogeneous Delays: Different students update at different frequencies
    "update_freqs": [1, 4], 
    
    # Iterated Learning: Reinitialize students at every X epochs to encourage continual adaptation
    "reinit_epochs": [5,10,15,20,25,30,35,40,45],
    
    # Teaching loss weight
    "teaching_lambda": 0.5
}

if __name__ == "__main__":
    # --- 2. Define Custom Folders ---
    # This keeps your experiments organized by run name
    lmb = TEACHING_CONFIG["teaching_lambda"]
    beta = TEACHING_CONFIG["beta"]
    EXPERIMENT_NAME = f"teaching_stud2_lmb{lmb}_b{beta}"
    print("EXPERIMENT_NAME:", EXPERIMENT_NAME)
    VIS_DIR = f"results/{EXPERIMENT_NAME}/visualizations"
    EVAL_DIR = f"results/{EXPERIMENT_NAME}/evaluation"

    train_dl, val_dl, test_dl, full_ds = make_dataloaders()
    
    model = LitTeachingVae(z_size=64, beta=beta, config=TEACHING_CONFIG)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss", 
        dirpath=f"checkpoints/{EXPERIMENT_NAME}/", # Good practice to separate checkpoints too
        filename="best_teaching_vae"
    )
    
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_callback, ImageLogger(save_dir=VIS_DIR)]
    )
    
    trainer.fit(model, train_dl, val_dl)
    run_final_eval(model, test_dl, full_ds, output_dir=EVAL_DIR)
    evaluate_linear_probe(model, train_dl, val_dl, test_dl, model.device)