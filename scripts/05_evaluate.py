import argparse
import sys
import yaml
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.patch_dataset import FloodPatchDataset
from src.models.unet import UNet

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation model on the Test split.")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to data config file")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/unet_best.pt", help="Path to model checkpoint")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    dataset_root = config["dataset_root"]
    patch_size = config["patch_size"]
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("Preparing test dataset...")
    test_dataset = FloodPatchDataset(
        dataset_root=dataset_root,
        split="Test",
        patch_size=patch_size,
        stride=patch_size, # non-overlapping for test metrics
        augment=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    
    # Load model
    print("Loading model and checkpoint...")
    model = UNet(in_channels=11, num_classes=6).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with Val Loss: {checkpoint['val_loss']:.4f}")
    
    # Track True Positives, False Positives, and False Negatives per class
    # Classes are 0 to 5.
    tp = np.zeros(6, dtype=np.int64)
    fp = np.zeros(6, dtype=np.int64)
    fn = np.zeros(6, dtype=np.int64)
    class_pixel_counts = np.zeros(6, dtype=np.int64)
    
    with torch.no_grad():
        for s2_patches, _, gt_patches in tqdm(test_loader, desc="Evaluating"):
            s2_patches = s2_patches.to(device)
            gt_patches = gt_patches.numpy() # Shape: (B, H, W)
            
            logits = model(s2_patches) # Shape: (B, 6, H, W)
            preds = torch.argmax(logits, dim=1).cpu().numpy() # Shape: (B, H, W)
            
            # Loop over batch elements and classes
            for b in range(preds.shape[0]):
                pred_b = preds[b]
                gt_b = gt_patches[b]
                
                # Valid mask (ignore labels >= 6)
                valid_mask = gt_b < 6
                
                for c in range(6):
                    pred_c = (pred_b == c) & valid_mask
                    gt_c = (gt_b == c) & valid_mask
                    
                    tp[c] += np.sum(pred_c & gt_c)
                    fp[c] += np.sum(pred_c & ~gt_c)
                    fn[c] += np.sum(~pred_c & gt_c)
                    class_pixel_counts[c] += np.sum(gt_c)
                    
    # Calculate Precision, Recall, and F1 per class
    print("\n" + "="*70)
    print(f"{'Class ID':<10}{'Class Name':<45}{'Pr (%)':<10}{'Rc (%)':<10}{'F1 (%)':<10}")
    print("="*70)
    
    class_names = [
        "Open surface water (2)",
        "Soil seriously affected (3)",
        "Soil moderately affected (4)",
        "Soil slightly affected (5)",
        "Vegetation affected (6)",
        "Not affected areas (7)"
    ]
    
    precision_list = []
    recall_list = []
    f1_list = []
    
    for c in range(6):
        pr = tp[c] / (tp[c] + fp[c] + 1e-10)
        rc = tp[c] / (tp[c] + fn[c] + 1e-10)
        f1 = 2 * pr * rc / (pr + rc + 1e-10)
        
        precision_list.append(pr)
        recall_list.append(rc)
        f1_list.append(f1)
        
        print(f"{c:<10}{class_names[c]:<45}{pr*100:<10.2f}{rc*100:<10.2f}{f1*100:<10.2f}")
        
    print("="*70)
    
    # Calculate pixel-weighted overall metrics
    total_valid_pixels = class_pixel_counts.sum()
    weights = class_pixel_counts / total_valid_pixels
    
    overall_pr = np.sum(np.array(precision_list) * weights)
    overall_rc = np.sum(np.array(recall_list) * weights)
    overall_f1 = np.sum(np.array(f1_list) * weights)
    
    print(f"{'ALL':<10}{'Overall (Pixel-Weighted)':<45}{overall_pr*100:<10.2f}{overall_rc*100:<10.2f}{overall_f1*100:<10.2f}")
    print("="*70)

if __name__ == "__main__":
    main()
