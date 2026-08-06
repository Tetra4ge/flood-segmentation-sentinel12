import sys
from pathlib import Path
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.patch_dataset import FloodPatchDataset

def test_loader():
    with open("configs/data.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    dataset_root = config["dataset_root"]
    patch_size = config["patch_size"]
    stride = config["stride"]
    
    print("Initializing train dataset...")
    train_dataset = FloodPatchDataset(
        dataset_root=dataset_root,
        split="Train",
        patch_size=patch_size,
        stride=stride,
        augment=True
    )
    
    print(f"Dataset length: {len(train_dataset)}")
    
    # Get first sample
    s2, s1, gt = train_dataset[0]
    print(f"S2 tensor shape: {s2.shape}, dtype: {s2.dtype}")
    print(f"S1 tensor shape: {s1.shape}, dtype: {s1.dtype}")
    print(f"GT tensor shape: {gt.shape}, dtype: {gt.dtype}")
    
    assert s2.shape == (11, 64, 64)
    assert s1.shape == (3, 64, 64)
    assert gt.shape == (64, 64)
    print("Dataset test passed successfully!")

if __name__ == "__main__":
    test_loader()
