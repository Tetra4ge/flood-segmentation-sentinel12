import argparse
import sys
import os
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.splits import get_scene_files
from src.data.raster_io import read_s2, read_s1, read_gt

def extract_and_save_patches(scenes, split, patch_size, stride, output_dir):
    split_out_dir = Path(output_dir) / split
    split_out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting patches for {split} split to {split_out_dir}...")
    patch_count = 0
    
    for scene in tqdm(scenes):
        name = scene["name"]
        
        # Load full scene arrays
        s2_img = read_s2(scene["S2"]) # (11, H, W)
        s1_img = read_s1(scene["S1"]) # (3, H, W)
        gt_img = read_gt(scene["GT"]) # (H, W)
        gt08_img = read_gt(scene["GT_0_8"]) # (H, W)
        
        _, H, W = s2_img.shape
        
        # Iterate sliding window
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                # Extract patches
                s2_patch = s2_img[:, y:y+patch_size, x:x+patch_size]
                s1_patch = s1_img[:, y:y+patch_size, x:x+patch_size]
                gt_patch = gt_img[y:y+patch_size, x:x+patch_size]
                gt08_patch = gt08_img[y:y+patch_size, x:x+patch_size]
                
                # Check if patch is completely empty (all zeros or outside data area).
                # S2 values are uint16, if the entire patch is 0 in all channels, it is nodata.
                if np.all(s2_patch == 0):
                    continue
                
                # Save patch to npz
                patch_filename = split_out_dir / f"{name}_y{y}_x{x}.npz"
                np.savez_compressed(
                    patch_filename,
                    s2=s2_patch.astype(np.float32),
                    s1=s1_patch.astype(np.float32),
                    gt=gt_patch.astype(np.uint8),
                    gt08=gt08_patch.astype(np.uint8)
                )
                patch_count += 1
                
    print(f"Extraction complete for {split}: {patch_count} patches created.")
    return patch_count

def main():
    parser = argparse.ArgumentParser(description="Extract and cache 64x64 patches from full scenes.")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to data config file")
    parser.add_argument("--stride", type=int, default=None, help="Override stride in config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dataset_root = config["dataset_root"]
    patch_size = config["patch_size"]
    stride = args.stride if args.stride is not None else config["stride"]
    output_dir = config["cache_root"]
    
    # We will use non-overlapping patches (stride = patch_size) for Test,
    # and we can use stride = 32 or 64 for Train to control data density.
    # By default, use the stride from configs/data.yaml.
    
    # Process Train split
    train_scenes = get_scene_files(dataset_root, "Train")
    extract_and_save_patches(train_scenes, "Train", patch_size, stride, output_dir)
    
    # Process Test split (use stride = patch_size for non-overlapping evaluation)
    test_scenes = get_scene_files(dataset_root, "Test")
    extract_and_save_patches(test_scenes, "Test", patch_size, patch_size, output_dir)

if __name__ == "__main__":
    main()
