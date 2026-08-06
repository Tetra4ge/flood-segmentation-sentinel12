import argparse
import sys
import yaml
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.splits import get_scene_files
from src.data.raster_io import read_s2, read_s1

def main():
    parser = argparse.ArgumentParser(description="Compute mean and std for normalization.")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to data config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dataset_root = config["dataset_root"]
    scenes = get_scene_files(dataset_root, "Train")
    
    print(f"Computing statistics over {len(scenes)} training scenes...")
    
    s2_means = []
    s2_stds = []
    s1_means = []
    s1_stds = []
    
    for scene in scenes:
        print(f"Reading {scene['name']}...")
        s2 = read_s2(scene["S2"]) # (11, H, W)
        s1 = read_s1(scene["S1"]) # (3, H, W)
        
        # Flatten spatial dimensions
        s2_flat = s2.reshape(11, -1)
        s1_flat = s1.reshape(3, -1)
        
        # Mask out completely zero pixels (nodata) for stats computation
        valid_mask = np.any(s2_flat != 0, axis=0)
        
        if np.any(valid_mask):
            s2_valid = s2_flat[:, valid_mask]
            s1_valid = s1_flat[:, valid_mask]
            
            s2_means.append(np.mean(s2_valid, axis=1))
            s2_stds.append(np.std(s2_valid, axis=1))
            
            s1_means.append(np.mean(s1_valid, axis=1))
            s1_stds.append(np.std(s1_valid, axis=1))
            
    s2_mean = np.mean(s2_means, axis=0)
    s2_std = np.mean(s2_stds, axis=0)
    s1_mean = np.mean(s1_means, axis=0)
    s1_std = np.mean(s1_stds, axis=0)
    
    print("\n=== Sentinel-2 Stats (11 bands) ===")
    print(f"Mean: {s2_mean.tolist()}")
    print(f"Std:  {s2_std.tolist()}")
    
    print("\n=== Sentinel-1 Stats (3 bands) ===")
    print(f"Mean: {s1_mean.tolist()}")
    print(f"Std:  {s1_std.tolist()}")

if __name__ == "__main__":
    main()
