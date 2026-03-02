import torch
import torchvision
import pytorch_lightning as pl
import numpy as np
import os
import sys

# Handle disent import safely
try:
    from disent.metrics import metric_dci
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from disent.metrics import metric_dci
    except ImportError:
        print("CRITICAL WARNING: Could not import 'disent'. DCI metric will fail.")
        metric_dci = None

class ImageLogger(pl.Callback):
    def __init__(self, save_dir="visualizations"):
        super().__init__()
        self.save_dir = save_dir

    def on_validation_epoch_end(self, trainer, pl_module):
        if hasattr(pl_module, 'sample_imgs'):
            x, x_recon = pl_module.sample_imgs
            
            # Grid: Top = Original, Bottom = Reconstruction
            grid = torchvision.utils.make_grid(torch.cat([x, x_recon], dim=0), nrow=8)
            
            os.makedirs(self.save_dir, exist_ok=True)
            path = os.path.join(self.save_dir, f"epoch_{trainer.current_epoch}.png")
            torchvision.utils.save_image(grid, path)

def _get_encoder_output(model, x):
    if hasattr(model, 'teacher_encoder'):
        return model.teacher_encoder(x)
    elif hasattr(model, 'encoder'):
        return model.encoder(x)
    else:
        raise AttributeError("Model missing 'teacher_encoder' or 'encoder'.")

# --- 1. The Core Traversal Function (Single Source of Truth) ---
def generate_traversals(model, test_loader, device, output_dir="visualizations", filename="traversals.png", n_latents=10, n_steps=9):
    """
    Generates latent traversals and saves them to disk.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    # Get one batch
    batch = next(iter(test_loader))
    x = model._get_x(batch)[0:1].to(device) 
    
    with torch.no_grad():
        out = _get_encoder_output(model, x)
        mu = out[0] if isinstance(out, (tuple, list)) else out
        
        traversal_grid = []
        actual_latents = min(n_latents, mu.shape[1])
        
        for i in range(actual_latents):
            # Create a sweep for dimension i
            z_sweep = mu.repeat(n_steps, 1)
            sweep_vals = torch.linspace(-3, 3, n_steps).to(device)
            z_sweep[:, i] = sweep_vals
            
            recons = model.decoder(z_sweep)
            traversal_grid.append(recons)
        
        # Grid: Rows = Latent Dims, Cols = Interpolation Steps
        grid = torchvision.utils.make_grid(torch.cat(traversal_grid), nrow=n_steps)
        
        save_path = os.path.join(output_dir, filename)
        torchvision.utils.save_image(grid, save_path)
        print(f"Traversals saved to: {save_path}")

# --- 2. The Final Eval Function (Calls the function above) ---
def run_final_eval(model, test_loader, full_dataset, output_dir="eval_results"):
    model.eval()
    device = model.device
    print(f"\n=== Final Evaluation (Output: {output_dir}) ===")
    os.makedirs(output_dir, exist_ok=True)
    
    # A. Pixel RMSE
    all_sq_errors = 0
    total_pixels = 0
    with torch.no_grad():
        for batch in test_loader:
            x = model._get_x(batch).to(device)
            out = _get_encoder_output(model, x)
            mu = out[0] if isinstance(out, (tuple, list)) else out
            x_recon = model.decoder(mu)
            
            x_px = (x * 255.0).clamp(0, 255)
            x_recon_px = (x_recon * 255.0).clamp(0, 255)
            all_sq_errors += (x_px - x_recon_px).pow(2).sum().item()
            total_pixels += torch.numel(x)

    rmse_px = np.sqrt(all_sq_errors / total_pixels)
    print(f"Pixel-Space RMSE: {rmse_px:.4f}")

    # B. DCI Disentanglement
    if metric_dci is not None:
        print("\n--- Calculating DCI (10k samples) ---")
        def get_mu(x):
            if x.shape[-1] == 3: x = x.permute(0, 3, 1, 2)
            if x.dtype == torch.uint8: x = x.float() / 255.0
            with torch.no_grad(): 
                out = _get_encoder_output(model, x.to(device))
                return out[0].cpu() if isinstance(out, tuple) else out.cpu()
        try:
            dci = metric_dci(full_dataset, get_mu, num_train=10000, num_test=2000, show_progress=True)  
            print(f"DCI Disentanglement: {dci['dci.disentanglement']:.4f}")
        except:
            print("Error calculating DCI. Skipping.")
    else:
        print("Skipping DCI.")

    # C. Latent Traversals (REUSE LOGIC)
    try:
        print("\n--- Generating Traversals ---")
        generate_traversals(
            model, 
            test_loader, 
            device, 
            output_dir=output_dir, 
            filename="final_traversals.png"
        )
    except:
        print("Error generating traversals. Skipping.")