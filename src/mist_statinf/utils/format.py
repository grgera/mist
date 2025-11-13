import numpy as np
import torch
import torch.nn.functional as F

def as_2col(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.ndim == 1: X = X.reshape(-1, 1)
    if Y.ndim == 1: Y = Y.reshape(-1, 1)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"Mismatched lengths: X={X.shape}, Y={Y.shape}")
    return np.concatenate([X, Y], axis=1) 

def pad_truncate(x: torch.Tensor, target_feature_dim: int) -> torch.Tensor:
    """
    x: (B, L, Dcur) -> return (B, L, target_feature_dim)
    - pad zeros if Dcur < target_feature_dim
    - truncate if Dcur > target_feature_dim
    """
    cur = x.shape[-1]
    if cur == target_feature_dim:
        return x
    if cur < target_feature_dim:
        pad = target_feature_dim - cur
        return F.pad(x, (0, pad), "constant", 0)
    return x[..., :target_feature_dim]