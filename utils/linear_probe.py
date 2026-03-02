import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Map Shapes3D indices to readable names
FACTOR_NAMES = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']

def extract_features(model, dataloader, device):
    """
    Helper to extract z (latents) and y (factors) from a dataloader.
    Returns: numpy arrays (N, z_dim) and (N, n_factors)
    """
    model.eval()
    z_list = []
    y_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Handle Tuple (x, y) vs Dict
            if isinstance(batch, dict):
                 # If you are using the raw DisentDataset without the wrapper
                x = batch['x_targ']
                # Try to retrieve labels if possible, otherwise skip or error
                if hasattr(dataloader.dataset, 'dataset') and hasattr(dataloader.dataset.dataset, 'gt_data'):
                     # This is tricky with random_split, usually easier if batch is a tuple
                     pass 
                else:
                    # Fallback if using the wrapper we defined earlier
                    pass
            else:
                x, y = batch
            
            x = x.to(device)
            
            # Use Teacher Encoder if available (Teaching VAE), else standard Encoder
            if hasattr(model, 'teacher_encoder'):
                mu, _ = model.teacher_encoder(x)
            else:
                mu, _ = model.encoder(x)
                
            z_list.append(mu.cpu().numpy())
            y_list.append(y.cpu().numpy())
            
    return np.concatenate(z_list, axis=0), np.concatenate(y_list, axis=0)

def evaluate_linear_probe(model, train_dl, val_dl, test_dl, device):
    """
    Trains a linear classifier on the Train set latents.
    Evaluates on Test set latents.
    """
    print("\n=== Ground Truth Factor Classification (Linear Probe) ===")
    
    # 1. Extract Features from ALL splits
    print("Extracting features from Train...")
    z_train, y_train = extract_features(model, train_dl, device)
    
    print("Extracting features from Val...")
    z_val, y_val = extract_features(model, val_dl, device)
    
    print("Extracting features from Test...")
    z_test, y_test = extract_features(model, test_dl, device)
    
    # 2. Standardize (Fit on TRAIN only to avoid leakage)
    scaler = StandardScaler()
    z_train = scaler.fit_transform(z_train)
    z_val = scaler.transform(z_val)   # Apply train statistics
    z_test = scaler.transform(z_test) # Apply train statistics
    
    # 3. Train & Eval Classifiers
    accuracies = {}
    print("-" * 65)
    print(f"{'Factor':<15} | {'Val Acc':<10} | {'Test Acc':<10}")
    print("-" * 65)
    
    # Iterate over each factor (Floor, Wall, etc.)
    for i, factor_name in enumerate(FACTOR_NAMES):
        # # --- SANITY CHECK: Random Labels ---
        # # Train a classifier on shuffled labels. It SHOULD fail.
        # y_train_shuffled = np.random.permutation(y_train[:, i])
        # clf_dummy = LogisticRegression(max_iter=100)
        # clf_dummy.fit(z_train, y_train_shuffled)
        # dummy_acc = accuracy_score(y_test[:, i], clf_dummy.predict(z_test))
        # print("EXAMPLE OF LABEL", y_test[:100, i])
        # print(f"DEBUG: {factor_name} Random Label Acc: {dummy_acc:.2%} (Should be ~10%)")
        # # -----------------------------------

        # Logistic Regression (Linear Probe)
        clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto', n_jobs=-1)
        
        # Train on Training Set
        clf.fit(z_train, y_train[:, i])
        
        # Evaluate on Validation (Good for checking convergence/overfitting)
        val_pred = clf.predict(z_val)
        val_acc = accuracy_score(y_val[:, i], val_pred)
        
        # Evaluate on Test (The real metric)
        test_pred = clf.predict(z_test)
        test_acc = accuracy_score(y_test[:, i], test_pred)
        
        accuracies[factor_name] = test_acc
        print(f"{factor_name:<15} | {val_acc:.2%}    | {test_acc:.2%}")
        
    avg_acc = np.mean(list(accuracies.values()))
    print("-" * 65)
    print(f"{'Average':<15} | {'':<10} | {avg_acc:.2%}")
    print("Note: High Test Accuracy (>90%) indicates the factor is linearly disentangled.")
    
    return accuracies