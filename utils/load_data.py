import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

class Shapes3DFast(Dataset):
    def __init__(self, root_dir="data/", filename="3dshapes_uncompressed.h5"):
        self.file_path = os.path.join(root_dir, filename)
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Please run preprocess.py first to create {filename}")

        # 1. Get Length
        with h5py.File(self.file_path, 'r') as f:
            self.length = f['images'].shape[0]

        # 2. GENERATE INTEGER LABELS MATHEMATICALLY
        # 3DShapes varies factors in this order (Slowest -> Fastest):
        # Floor(10), Wall(10), Object(10), Scale(8), Shape(4), Orientation(15)
        # We compute the strides to convert linear index -> factor indices
        
        print("Generating integer labels from indices...")
        factor_sizes = [10, 10, 10, 8, 4, 15]
        
        # Calculate strides: [48000, 4800, 480, 60, 15, 1]
        strides = np.cumprod([1] + factor_sizes[::-1])[:-1][::-1]
        
        # Vectorized label generation (Much faster than loops)
        # Create a range [0, 1, ..., 479999]
        indices = torch.arange(self.length, dtype=torch.long)
        
        labels = []
        for stride in strides:
            # Integer Division gives the index for this factor
            labels.append((indices // stride) % 100) # % 100 is just safety, technically not needed
            # Update indices for next factor (modulo)
            indices = indices % stride
            
        # Stack to shape (N, 6) -> [[0,0,0,0,0,0], [0,0,0,0,0,1], ...]
        self.labels = torch.stack(labels, dim=1)

        self.images = None
        self.file = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.file_path, 'r', swmr=True, libver='latest')
            self.images = self.file['images']

        img_np = self.images[idx]
        x = torch.from_numpy(img_np).float().div(255.0).permute(2, 0, 1)

        return x, self.labels[idx]

    # Add back the compatibility method for DCI if needed
    def dataset_sample_batch_with_factors(self, num_samples, mode='input'):
        indices = np.random.randint(0, len(self), size=num_samples)
        batch_x = []
        batch_y = []
        for idx in indices:
            x, y = self[idx]
            batch_x.append(x.unsqueeze(0))
            batch_y.append(y.unsqueeze(0))
        return torch.cat(batch_x, dim=0), torch.cat(batch_y, dim=0)

def make_dataloaders(data_root="data/", batch_size=64):
    # USE THE FAST DATASET
    dataset = Shapes3DFast(root_dir=data_root)
    
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len
    
    train_set, val_set, test_set = random_split(
        dataset, [train_len, val_len, test_len], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # 4 workers is usually optimal for SSDs
    num_workers = 4 
    
    kwargs = {
        'num_workers': num_workers, 
        'pin_memory': True,
        'persistent_workers': True if num_workers > 0 else False
    } if torch.cuda.is_available() else {}
    
    return (DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs),
            DataLoader(val_set, batch_size=batch_size, shuffle=False, **kwargs),
            DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs),
            dataset)

def make_challenging_dataloaders(data_root="data/", batch_size=64):
    dataset = Shapes3DFast(root_dir=data_root)
    labels = dataset.labels  # (480000, 6)
    
    # Define Factor Map for reference:
    # 2: object_hue (0:Red, 2:Green, 5:Blue, etc. - based on 10 steps)
    # 4: shape (0:Cube, 1:Cylinder, 2:Sphere, 3:Pill)

    # Define our "Forbidden" combinations (Object Hue, Shape)
    # You can customize these indices based on the 3DShapes factor values
    forbidden_combinations = [
        (0, 0), # Red Cube
        (2, 1), # Green Cylinder
        (5, 2), # Blue Sphere
        (8, 3), # Magenta Pill
    ]

    # Initialize a mask of False (nothing is forbidden yet)
    is_forbidden = np.zeros(len(dataset), dtype=bool)

    for hue_idx, shape_idx in forbidden_combinations:
        # Create mask for this specific intersection
        match = (labels[:, 2] == hue_idx) & (labels[:, 4] == shape_idx)
        is_forbidden = is_forbidden | match.numpy()

    all_indices = np.arange(len(dataset))
    test_indices = all_indices[is_forbidden]      # ONLY the challenging hold-outs
    train_val_candidates = all_indices[~is_forbidden] # Everything else

    # Split candidates into Train and Validation (90/10)
    np.random.seed(42)
    np.random.shuffle(train_val_candidates)
    split = int(0.9 * len(train_val_candidates))
    
    train_idx = train_val_candidates[:split]
    val_idx = train_val_candidates[split:]
    
    # Subsets
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)
    test_set = torch.utils.data.Subset(dataset, test_indices)

    print(f"\n--- CHALLENGING SPLIT ---")
    print(f"Banned combinations: {forbidden_combinations}")
    print(f"Training samples: {len(train_set)}")
    print(f"Test samples (OOD): {len(test_set)}")

    kwargs = {'num_workers': 4, 'pin_memory': True, 'persistent_workers': True}
    return (DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs),
            DataLoader(val_set, batch_size=batch_size, shuffle=False, **kwargs),
            DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs),
            dataset)