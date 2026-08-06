import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneralizedDiceLoss(nn.Module):
    def __init__(self, ignore_index: int = 6, num_classes: int = 6, eps: float = 1e-6):
        """
        Generalized Dice Loss (GDL) with class-weighting and support for ignore index.
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, num_classes, H, W) float tensor
            targets: (B, H, W) int64 tensor containing class labels
        """
        # Softmax over channels to get probabilities
        probs = F.softmax(logits, dim=1) # (B, num_classes, H, W)
        
        # Mask out pixels that correspond to ignore classes (e.g. >= ignore_index)
        # In our case, ignore classes are 6 (cloud) and 7 (shadow).
        # We can construct a binary mask of valid pixels.
        valid_mask = (targets < self.ignore_index).unsqueeze(1) # (B, 1, H, W)
        
        # Zero out ignore pixels in probs and targets
        probs = probs * valid_mask
        
        # Convert targets to one-hot encoding
        # Clamp targets to avoid index errors in one_hot before masking
        targets_clamped = torch.clamp(targets, 0, self.num_classes - 1)
        targets_onehot = F.one_hot(targets_clamped, num_classes=self.num_classes) # (B, H, W, num_classes)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float() # (B, num_classes, H, W)
        targets_onehot = targets_onehot * valid_mask
        
        # Sum over spatial dimensions (B, num_classes)
        intersection = torch.sum(probs * targets_onehot, dim=(0, 2, 3))
        union = torch.sum(probs + targets_onehot, dim=(0, 2, 3))
        
        # Class weights set inversely proportional to square of class frequency
        class_pixel_counts = torch.sum(targets_onehot, dim=(0, 2, 3))
        weights = 1.0 / (class_pixel_counts ** 2 + self.eps)
        
        # Normalize weights to prevent underflow/overflow
        weights = weights / (torch.sum(weights) + self.eps)
        
        # Compute GDL
        numerator = 2.0 * torch.sum(weights * intersection)
        denominator = torch.sum(weights * union)
        
        gdl = 1.0 - (numerator + self.eps) / (denominator + self.eps)
        return gdl
