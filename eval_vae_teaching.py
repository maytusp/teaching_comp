import os
import torch
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Import your modules
from models import LitTeachingVae
from utils.load_data import make_dataloaders
from utils.eval_recon import run_final_eval
from utils.linear_probe import evaluate_linear_probe 

# --- CONFIGURATION (Must match training config if not saved in hparams) ---
# If your checkpoint has hparams saved, LitTeachingVae will load them automatically.
# But if you need to force a config, define it here.
TEACHING_CONFIG = {
    "n_students": 5,
    "student_noise_std": 0.1,
    "update_freqs": [1, 2, 3, 4, 5], 
    "reinit_epochs": [10, 20, 30, 40],
    "teaching_lambda": 0.5
}

def load_and_evaluate(checkpoint_path, batch_size=64, device="cuda"):
    print(f"\n=== Loading Checkpoint: {checkpoint_path} ===")
    
    # 1. Load Data
    # We need the full dataset for DCI and splits for Linear Probe
    print("Loading Data...")
    train_dl, val_dl, test_dl, full_ds = make_dataloaders(batch_size=batch_size)
    
    # 2. Load Model
    # We load from checkpoint. We must provide the config if it wasn't saved in hparams,
    # or if we want to override it.
    print("Loading Model...")
    model = LitTeachingVae.load_from_checkpoint(
        checkpoint_path, 
        z_size=64,       # Must match training
        beta=1.0,        # Must match training
        config=TEACHING_CONFIG
    )
    
    model.eval()
    model.to(device)
    
    # Create output directory based on checkpoint name
    ckpt_name = os.path.basename(checkpoint_path).replace('.ckpt', '')
    parent_dir = os.path.dirname(os.path.dirname(checkpoint_path)) # Go up two levels
    eval_dir = os.path.join(parent_dir, "evaluation_post_hoc", ckpt_name)
    os.makedirs(eval_dir, exist_ok=True)
    
    print(f"Saving results to: {eval_dir}")

    # -------------------------------------------------
    # 3. Run Evaluations
    # -------------------------------------------------

    # A. Reconstruction & DCI (Standard Metrics)
    print("\n[1/3] Running Reconstruction & DCI Eval...")
    run_final_eval(model, test_dl, full_ds, output_dir=eval_dir)

    # B. Linear Probe (Compositionality Metric)
    print("\n[2/3] Running Linear Probe (Factor Classification)...")
    # This checks if factors are linearly separable (disentangled)
    accuracies = evaluate_linear_probe(model, train_dl, val_dl, test_dl, device)
    
    # Save Probe Results to text file
    with open(os.path.join(eval_dir, "linear_probe_results.txt"), "w") as f:
        f.write("Factor Classification Accuracy (Linear Probe)\n")
        f.write("=============================================\n")
        for factor, acc in accuracies.items():
            f.write(f"{factor:<15}: {acc:.4f}\n")
        avg_acc = sum(accuracies.values()) / len(accuracies)
        f.write("---------------------------------------------\n")
        f.write(f"{'Average':<15}: {avg_acc:.4f}\n")
    
    print(f"\nDone! All results saved to {eval_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained VAE model")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the .ckpt file")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for evaluation")
    
    args = parser.parse_args()
    
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    
    load_and_evaluate(args.ckpt, device=device)