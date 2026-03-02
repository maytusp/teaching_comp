import h5py
import numpy as np
import os
from tqdm import tqdm

def uncompress_3dshapes(root_dir="data/", src_file="3dshapes.h5", dest_file="3dshapes_uncompressed.h5"):
    src_path = os.path.join(root_dir, src_file)
    dest_path = os.path.join(root_dir, dest_file)
    
    if not os.path.exists(src_path):
        # check shapes3d subdir
        src_path = os.path.join(root_dir, "shapes3d", src_file)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Could not find {src_file}")

    print(f"Processing {src_path} -> {dest_path}")
    print("This will take a minute but will make training 100x faster.")

    with h5py.File(src_path, 'r') as f_src:
        # Get shapes
        num_samples = f_src['images'].shape[0]
        img_shape = f_src['images'].shape[1:] # (64, 64, 3)
        lbl_shape = f_src['labels'].shape[1:] # (6,)
        
        with h5py.File(dest_path, 'w') as f_dest:
            # Create datasets without compression
            # chunks=(1, 64, 64, 3) optimizes for reading one image at a time
            dset_img = f_dest.create_dataset(
                'images', 
                shape=(num_samples, *img_shape), 
                dtype='uint8', 
                chunks=(1, 64, 64, 3) 
            )
            dset_lbl = f_dest.create_dataset(
                'labels', 
                shape=(num_samples, *lbl_shape), 
                dtype='float32'
            )
            
            # Copy data in batches to be memory efficient
            batch_size = 1000
            for i in tqdm(range(0, num_samples, batch_size)):
                end = min(i + batch_size, num_samples)
                dset_img[i:end] = f_src['images'][i:end]
                dset_lbl[i:end] = f_src['labels'][i:end]

    print(f"Done! Created {dest_path}")
    print(f"Original size: {os.path.getsize(src_path) / 1e6:.2f} MB")
    print(f"New size:      {os.path.getsize(dest_path) / 1e9:.2f} GB")

if __name__ == "__main__":
    uncompress_3dshapes()