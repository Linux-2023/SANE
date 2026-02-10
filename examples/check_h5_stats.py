import h5py
import numpy as np
import sys

def check_dataset_stats(h5_path):
    print(f"Checking stats for {h5_path}...")
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'features' not in f and 'image_feats' not in f:
                print("Error: No 'features' dataset found.")
                return

            feat_key = 'features' if 'features' in f else 'image_feats'
            ds = f[feat_key]
            print(f"Features shape: {ds.shape}, dtype: {ds.dtype}")
            
            # Read a chunk to avoid out of memory if huge
            # Read first 1000 or all
            limit = min(2000, ds.shape[0])
            data = ds[:limit]
            
            # Compute stats
            print(f"\n--- Statistics (First {limit} samples) ---")
            print(f"Mean: {np.mean(data):.4f}")
            print(f"Std:  {np.std(data):.4f}")
            print(f"Min:  {np.min(data):.4f}")
            print(f"Max:  {np.max(data):.4f}")
            
            # Check latents as well just in case
            if 'latents' in f:
                latents = f['latents'][:limit]
                print(f"\n--- Latents Stats ---")
                print(f"Mean: {np.mean(latents):.4f}")
                print(f"Std:  {np.std(latents):.4f}")
                print(f"Max:  {np.max(latents):.4f}")

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    path = "latent_dataset.h5"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    check_dataset_stats(path)
