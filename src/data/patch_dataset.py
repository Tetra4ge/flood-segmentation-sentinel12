import torch
from torch.utils.data import Dataset
import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

from src.data.splits import get_scene_files

# Computed from training set:
S2_MEAN = np.array([0.0436, 0.0557, 0.0723, 0.0838, 0.1151, 0.1598, 0.1800, 0.1974, 0.2004, 0.2233, 0.1706], dtype=np.float32)
S2_STD  = np.array([0.0182, 0.0233, 0.0272, 0.0340, 0.0380, 0.0598, 0.0686, 0.0751, 0.0728, 0.0674, 0.0606], dtype=np.float32)

S1_MEAN = np.array([-21.6081, -13.6686, -8.0403], dtype=np.float32)
S1_STD  = np.array([3.4706, 3.0644, 2.5309], dtype=np.float32)

class FloodPatchDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        split: str,
        patch_size: int = 64,
        stride: int = 64,
        augment: bool = False,
        cache_in_memory: bool = False,
        load_s1: bool = True
    ):
        """
        PyTorch Dataset for Sentinel-1/2 Flood Segmentation using windowed reads or in-memory cache.
        """
        self.dataset_root = dataset_root
        self.split = split
        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment
        self.cache_in_memory = cache_in_memory
        self.load_s1 = load_s1
        
        # Get aligned scene files
        self.scenes = get_scene_files(dataset_root, split)
        
        self.s2_readers = []
        self.s1_readers = []
        self.gt_readers = []
        
        self.s2_cache = {}
        self.s1_cache = {}
        self.gt_cache = {}
        
        self.patches = []
        
        print(f"Initializing {split} dataset (cache_in_memory={cache_in_memory}, load_s1={load_s1})...")
        
        for scene_idx, scene in enumerate(self.scenes):
            # Open temporary readers to read metadata/dimensions
            with rasterio.open(scene["S2"]) as s2_src:
                H, W = s2_src.height, s2_src.width
                
            # If cache_in_memory is True, read full arrays now
            if cache_in_memory:
                # S2
                with rasterio.open(scene["S2"]) as s2_src:
                    s2_img = s2_src.read().astype(np.float32)
                    s2_img /= 10000.0
                    self.s2_cache[scene_idx] = s2_img
                # S1
                if load_s1:
                    with rasterio.open(scene["S1"]) as s1_src:
                        self.s1_cache[scene_idx] = s1_src.read().astype(np.float32)
                # GT
                with rasterio.open(scene["GT_0_8"]) as gt_src:
                    self.gt_cache[scene_idx] = gt_src.read(1).astype(np.int64)
            else:
                # Store persistent readers for lazy window reading
                self.s2_readers.append(rasterio.open(scene["S2"]))
                if load_s1:
                    self.s1_readers.append(rasterio.open(scene["S1"]))
                self.gt_readers.append(rasterio.open(scene["GT_0_8"]))
            
            # Generate grid window coordinates
            for y in range(0, H - patch_size + 1, stride):
                for x in range(0, W - patch_size + 1, stride):
                    self.patches.append({
                        "scene_idx": scene_idx,
                        "y": y,
                        "x": x
                    })
                    
        print(f"Loaded {split} dataset with {len(self.patches)} total patches from {len(self.scenes)} scenes.")

    def close(self):
        """Close all open rasterio dataset readers."""
        for r in self.s2_readers:
            try: r.close()
            except: pass
        for r in self.s1_readers:
            try: r.close()
            except: pass
        for r in self.gt_readers:
            try: r.close()
            except: pass

    def __del__(self):
        self.close()

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patch_info = self.patches[idx]
        scene_idx = patch_info["scene_idx"]
        y, x = patch_info["y"], patch_info["x"]
        
        if self.cache_in_memory:
            # Read from memory cache (extremely fast!)
            s2_data = self.s2_cache[scene_idx][:, y:y+self.patch_size, x:x+self.patch_size].copy()
            if self.load_s1:
                s1_data = self.s1_cache[scene_idx][:, y:y+self.patch_size, x:x+self.patch_size].copy()
            else:
                s1_data = np.zeros((3, self.patch_size, self.patch_size), dtype=np.float32)
            gt_data = self.gt_cache[scene_idx][y:y+self.patch_size, x:x+self.patch_size].copy()
        else:
            # Read window from open readers
            window = Window(x, y, self.patch_size, self.patch_size)
            s2_data = self.s2_readers[scene_idx].read(window=window).astype(np.float32)
            s2_data /= 10000.0
            
            if self.load_s1:
                s1_data = self.s1_readers[scene_idx].read(window=window).astype(np.float32)
            else:
                s1_data = np.zeros((3, self.patch_size, self.patch_size), dtype=np.float32)
                
            gt_data = self.gt_readers[scene_idx].read(1, window=window).astype(np.int64)
            
        # Standardize features using training stats
        for c in range(11):
            s2_data[c] = (s2_data[c] - S2_MEAN[c]) / S2_STD[c]
            
        if self.load_s1:
            for c in range(3):
                s1_data[c] = (s1_data[c] - S1_MEAN[c]) / S1_STD[c]
            
        # Data augmentation (only for Train split)
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                s2_data = np.flip(s2_data, axis=2).copy()
                if self.load_s1:
                    s1_data = np.flip(s1_data, axis=2).copy()
                gt_data = np.flip(gt_data, axis=1).copy()
                
            # Random vertical flip
            if np.random.rand() > 0.5:
                s2_data = np.flip(s2_data, axis=1).copy()
                if self.load_s1:
                    s1_data = np.flip(s1_data, axis=1).copy()
                gt_data = np.flip(gt_data, axis=0).copy()
                
            # Random 90-degree rotations
            rot_k = np.random.randint(0, 4)
            if rot_k > 0:
                s2_data = np.rot90(s2_data, k=rot_k, axes=(1, 2)).copy()
                if self.load_s1:
                    s1_data = np.rot90(s1_data, k=rot_k, axes=(1, 2)).copy()
                gt_data = np.rot90(gt_data, k=rot_k, axes=(0, 1)).copy()
                
        # Convert to tensors
        s2_tensor = torch.from_numpy(s2_data)
        s1_tensor = torch.from_numpy(s1_data)
        gt_tensor = torch.from_numpy(gt_data)
        
        return s2_tensor, s1_tensor, gt_tensor
