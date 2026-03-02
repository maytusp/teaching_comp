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

        # 1. Load Labels into RAM (Fast)
        with h5py.File(self.file_path, 'r') as f:
            self.length = f['images'].shape[0]
            self.labels = torch.from_numpy(f['labels'][:]).long()

        self.images = None
        self.file = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.file is None:
            # swmr=True and libver='latest' are faster for concurrent reads
            self.file = h5py.File(self.file_path, 'r', swmr=True, libver='latest')
            self.images = self.file['images']

        # 2. Read Image (Instant because it's uncompressed)
        img_np = self.images[idx]
        
        # 3. Convert
        x = torch.from_numpy(img_np).float().div(255.0)
        x = x.permute(2, 0, 1)

        return x, self.labels[idx]

    def dataset_sample_batch_with_factors(self, num_samples, mode='input'):
        indices = np.random.randint(0, len(self), size=num_samples)
        
        # Fast fetch loop
        batch_x = []
        batch_y = []
        
        # We fetch one by one. Since HDF5 is uncompressed now, this is reasonably fast.
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