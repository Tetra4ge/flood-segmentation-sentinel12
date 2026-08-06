import rasterio
import numpy as np

def read_s2(path: str) -> np.ndarray:
    """
    Read Sentinel-2 GeoTIFF file.
    Returns:
        np.ndarray of shape (11, H, W) normalized to [0, 1.0] reflectance scale.
    """
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        # Normalization: Sentinel-2 BOA reflectance values are scaled by 10000.
        data /= 10000.0
        return data

def read_s1(path: str) -> np.ndarray:
    """
    Read Sentinel-1 GeoTIFF file.
    Returns:
        np.ndarray of shape (3, H, W) containing VV, VH, and ratio in dB.
    """
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        return data

def read_gt(path: str) -> np.ndarray:
    """
    Read Ground Truth GeoTIFF file.
    Returns:
        np.ndarray of shape (H, W) containing class labels.
    """
    with rasterio.open(path) as src:
        data = src.read(1) # read band 1
        return data
