import argparse
import sys
import yaml
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.splits import get_scene_files
from src.data.raster_io import read_gt

def main():
    parser = argparse.ArgumentParser(description="Calculate class pixel statistics in Train/Test splits.")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to data config file")
    parser.add_argument("--out", default="outputs/class_stats.csv", help="Path to save output CSV stats")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dataset_root = config["dataset_root"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    class_names = {
        2: "Open surface water",
        3: "Soil seriously affected by waterlogging",
        4: "Soil moderately affected by waterlogging",
        5: "Soil slightly affected by waterlogging",
        6: "Vegetation affected by waterlogging",
        7: "Not affected areas",
        10: "Cloud / missing data",
        11: "Shadows"
    }
    
    mapping = {
        2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 10: 6, 11: 7
    }

    stats_list = []

    for split in ["Train", "Test"]:
        print(f"Analyzing {split} split...")
        try:
            scenes = get_scene_files(dataset_root, split)
        except Exception as e:
            print(f"Error loading {split} scenes: {e}")
            continue

        for scene in scenes:
            name = scene["name"]
            gt = read_gt(scene["GT"])
            
            # Count pixels
            unique, counts = np.unique(gt, return_counts=True)
            counts_dict = dict(zip(unique, counts))
            
            # Record statistics
            for val in [2, 3, 4, 5, 6, 7, 10, 11]:
                pixel_count = counts_dict.get(val, 0)
                stats_list.append({
                    "Split": split,
                    "Scene": name,
                    "Class_ID": val,
                    "Mapped_ID": mapping[val],
                    "Class_Name": class_names[val],
                    "Pixel_Count": pixel_count
                })

    df = pd.DataFrame(stats_list)
    df.to_csv(args.out, index=False)
    print(f"\nStatistics saved to: {args.out}")

    # Display summary of total pixels per class per split
    summary = df.groupby(["Split", "Class_ID", "Mapped_ID", "Class_Name"])["Pixel_Count"].sum().reset_index()
    print("\n=== Dataset Summary ===")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
