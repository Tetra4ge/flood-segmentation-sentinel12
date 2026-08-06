import argparse
import os
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.patch_dataset import FloodPatchDataset
from src.models.unet import UNet
from src.losses.generalized_dice import GeneralizedDiceLoss
from src.losses.weighted_focal import WeightedFocalLoss

# Pixel counts from training set (used for class weighting in focal loss)
TRAIN_PIXEL_COUNTS = [7674525, 651441, 900550, 4493763, 3450129, 195012790]

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_class_weights():
    # Inverse frequency weights: total / (num_classes * count)
    counts = np.array(TRAIN_PIXEL_COUNTS, dtype=np.float32)
    total = counts.sum()
    weights = total / (len(counts) * counts)
    # Normalize weights so they average to 1.0
    weights = weights / weights.mean()
    return torch.from_numpy(weights).float()

def main():
    parser = argparse.ArgumentParser(description="Train U-Net baseline on S2 only.")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to data config file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-dir", default="outputs/checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--stride", type=int, default=None, help="Override training stride")
    args = parser.parse_args()

    set_seed(args.seed)
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    dataset_root = config["dataset_root"]
    patch_size = config["patch_size"]
    stride = args.stride if args.stride is not None else config["stride"]
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Instantiate datasets and dataloaders
    print("Preparing datasets...")
    train_dataset = FloodPatchDataset(
        dataset_root=dataset_root,
        split="Train",
        patch_size=patch_size,
        stride=stride,
        augment=True
    )
    
    test_dataset = FloodPatchDataset(
        dataset_root=dataset_root,
        split="Test",
        patch_size=patch_size,
        stride=patch_size, # non-overlapping for validation/test
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False
    )
    
    # Instantiate Model
    print("Initializing model...")
    model = UNet(in_channels=11, num_classes=6).to(device)
    
    # Loss functions
    class_weights = get_class_weights()
    print(f"Computed focal loss class weights: {class_weights.tolist()}")
    
    criterion_gdl = GeneralizedDiceLoss(ignore_index=6, num_classes=6)
    criterion_wfl = WeightedFocalLoss(alpha=class_weights, gamma=2.0, ignore_index=6)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_loss = float("inf")
    
    # Training Loop
    print("\nStarting training loop...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_gdl = 0.0
        train_wfl = 0.0
        
        # We loop over the batches
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for batch_idx, (s2_patches, _, gt_patches) in enumerate(loop):
            s2_patches = s2_patches.to(device) # (B, 11, 64, 64)
            gt_patches = gt_patches.to(device) # (B, 64, 64)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(s2_patches) # (B, 6, 64, 64)
            
            # Losses
            loss_gdl = criterion_gdl(logits, gt_patches)
            loss_wfl = criterion_wfl(logits, gt_patches)
            loss = loss_gdl + loss_wfl
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_gdl += loss_gdl.item()
            train_wfl += loss_wfl.item()
            
            # Update tqdm progress bar
            loop.set_postfix({
                "loss": f"{loss.item():.4f}",
                "gdl": f"{loss_gdl.item():.4f}",
                "wfl": f"{loss_wfl.item():.4f}"
            })
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_gdl = train_gdl / len(train_loader)
        avg_train_wfl = train_wfl / len(train_loader)
        
        # Validation Loop (evaluate loss on Test set)
        model.eval()
        val_loss = 0.0
        val_gdl = 0.0
        val_wfl = 0.0
        
        with torch.no_grad():
            for s2_patches, _, gt_patches in test_loader:
                s2_patches = s2_patches.to(device)
                gt_patches = gt_patches.to(device)
                
                logits = model(s2_patches)
                
                loss_gdl = criterion_gdl(logits, gt_patches)
                loss_wfl = criterion_wfl(logits, gt_patches)
                loss = loss_gdl + loss_wfl
                
                val_loss += loss.item()
                val_gdl += loss_gdl.item()
                val_wfl += loss_wfl.item()
                
        avg_val_loss = val_loss / len(test_loader)
        avg_val_gdl = val_gdl / len(test_loader)
        avg_val_wfl = val_wfl / len(test_loader)
        
        print(f"Epoch {epoch}/{args.epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} (GDL: {avg_train_gdl:.4f}, WFL: {avg_train_wfl:.4f})")
        print(f"  Val Loss:   {avg_val_loss:.4f} (GDL: {avg_val_gdl:.4f}, WFL: {avg_val_wfl:.4f})")
        
        # Save checkpoint if it's the best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(args.save_dir, "unet_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f"  *** Best checkpoint saved to {checkpoint_path} (Val Loss: {avg_val_loss:.4f}) ***")
            
if __name__ == "__main__":
    main()
