import argparse
import sys
import os
import yaml
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.splits import get_scene_files
from src.data.raster_io import read_s2, read_s1, read_gt
import rasterio

def main():
    parser = argparse.ArgumentParser(description="Inspect dataset shapes, alignments, and metadata.")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to data config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dataset_root = config["dataset_root"]
    print(f"Dataset root: {dataset_root}")

    for split in ["Train", "Test"]:
        print(f"\n=================== INSPECTING {split.upper()} SPLIT ===================")
        try:
            scenes = get_scene_files(dataset_root, split)
        except Exception as e:
            print(f"Error loading {split} scenes: {e}")
            continue

        print(f"Found {len(scenes)} scenes.")
        for idx, scene in enumerate(scenes):
            name = scene["name"]
            print(f"\nScene {idx+1}/{len(scenes)}: {name}")
            
            # Check file sizes
            s1_size = os.path.getsize(scene["S1"]) / (1024 * 1024)
            s2_size = os.path.getsize(scene["S2"]) / (1024 * 1024)
            gt_size = os.path.getsize(scene["GT"]) / (1024 * 1024)
            gt08_size = os.path.getsize(scene["GT_0_8"]) / (1024 * 1024)
            print(f"  Files: S1({s1_size:.1f}MB) S2({s2_size:.1f}MB) GT({gt_size:.1f}MB) GT_0_8({gt08_size:.1f}MB)")

            # Check shapes via rasterio metadata without loading full data into memory
            with rasterio.open(scene["S2"]) as s2_src:
                s2_shape = (s2_src.count, s2_src.height, s2_src.width)
                s2_crs = s2_src.crs
                s2_bounds = s2_src.bounds
            
            with rasterio.open(scene["S1"]) as s1_src:
                s1_shape = (s1_src.count, s1_src.height, s1_src.width)
                s1_crs = s1_src.crs
                s1_bounds = s1_src.bounds

            with rasterio.open(scene["GT"]) as gt_src:
                gt_shape = (gt_src.count, gt_src.height, gt_src.width)
                gt_crs = gt_src.crs
                gt_bounds = gt_src.bounds

            with rasterio.open(scene["GT_0_8"]) as gt08_src:
                gt08_shape = (gt08_src.count, gt08_src.height, gt08_src.width)
                gt08_crs = gt08_src.crs
                gt08_bounds = gt08_src.bounds

            print(f"  S2 shape: {s2_shape}, CRS: {s2_crs}")
            print(f"  S1 shape: {s1_shape}, CRS: {s1_crs}")
            print(f"  GT shape: {gt_shape}, CRS: {gt_crs}")
            print(f"  GT_0_8 shape: {gt08_shape}, CRS: {gt08_crs}")

            # Verify shapes match exactly
            shape_match = (s2_shape[1:] == s1_shape[1:] == gt_shape[1:] == gt08_shape[1:])
            print(f"  Shape Match (H, W): {shape_match}")
            if not shape_match:
                print("  WARNING: Shape mismatch detected!")

            # Verify spatial bounds and CRS match exactly
            crs_match = (s2_crs == s1_crs == gt_crs == gt08_crs)
            bounds_match = (s2_bounds == s1_bounds == gt_bounds == gt08_bounds)
            print(f"  CRS Match: {crs_match}, Bounds Match: {bounds_match}")

            # Inspect actual values of GT
            gt_data = read_gt(scene["GT"])
            gt08_data = read_gt(scene["GT_0_8"])
            
            unique_gt = np.unique(gt_data)
            unique_gt08 = np.unique(gt08_data)
            print(f"  GT unique labels: {unique_gt.tolist()}")
            print(f"  GT_0_8 unique labels: {unique_gt08.tolist()}")

if __name__ == "__main__":
    main()
