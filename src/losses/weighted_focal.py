import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, ignore_index: int = 6):
        """
        Class-Weighted Focal Loss.
        Args:
            alpha: torch.Tensor of shape (num_classes,) containing weights for each class.
            gamma: float focal modulation factor (gamma >= 0).
            ignore_index: labels >= ignore_index are masked out of loss.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) float tensor
            targets: (B, H, W) int64 tensor
        """
        C = logits.size(1)
        
        # Reshape to (N, C) and (N,)
        logits_flat = logits.permute(0, 2, 3, 1).contiguous().view(-1, C)
        targets_flat = targets.view(-1)
        
        # Valid mask
        valid_mask = targets_flat < self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        logits_valid = logits_flat[valid_mask]
        targets_valid = targets_flat[valid_mask]
        
        # Compute soft log-probabilities
        log_probs = F.log_softmax(logits_valid, dim=1)
        
        # Get log probs for the target class
        log_probs_target = log_probs[torch.arange(len(targets_valid)), targets_valid]
        probs_target = torch.exp(log_probs_target)
        
        # Compute focal loss term: - (1 - p_t)^gamma * log(p_t)
        focal_weight = (1.0 - probs_target) ** self.gamma
        loss = - focal_weight * log_probs_target
        
        # Apply class weights if available
        if self.alpha is not None:
            # Send to same device
            self.alpha = self.alpha.to(logits.device)
            class_weights = self.alpha[targets_valid]
            loss = loss * class_weights
            
        return loss.mean()
