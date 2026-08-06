import os
from pathlib import Path
from typing import List, Dict

def get_scene_files(dataset_root: str, split: str) -> List[Dict[str, str]]:
    """
    Find and align S1, S2, GT, and GT_0_8 files for a given split (Train/Test).
    Returns a list of dicts: {'name': '...', 'S1': '...', 'S2': '...', 'GT': '...', 'GT_0_8': '...'}
    """
    root = Path(dataset_root)
    split_dir = root / "tifs" / split
    
    s1_dir = split_dir / "S1"
    s2_dir = split_dir / "S2"
    gt_dir = split_dir / "GT"
    gt08_dir = split_dir / "GT_0_8"
    
    # We use S2 files as reference
    s2_files = sorted(list(s2_dir.glob("*.tif")))
    
    aligned = []
    for s2_file in s2_files:
        name = s2_file.name
        
        s1_path = s1_dir / name
        gt_path = gt_dir / name
        gt08_path = gt08_dir / name
        
        if not s1_path.exists():
            raise FileNotFoundError(f"Missing S1 file: {s1_path}")
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing GT file: {gt_path}")
        if not gt08_path.exists():
            raise FileNotFoundError(f"Missing GT_0_8 file: {gt08_path}")
            
        aligned.append({
            'name': s2_file.stem,
            'S1': str(s1_path),
            'S2': str(s2_file),
            'GT': str(gt_path),
            'GT_0_8': str(gt08_path)
        })
        
    return aligned
