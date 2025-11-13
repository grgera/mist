import torch

def pinball_loss(y_pred: torch.Tensor, y_true: torch.Tensor, tau: torch.Tensor):
    """
    y_pred: (B, 1) or (B,)
    y_true: (B, 1) or (B,)
    tau: (B, 1) or (B,) with values in (0,1)
    returns scalar loss (mean over batch)
    """
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    tau = tau.view(-1)

    diff = y_true - y_pred
    loss = torch.max((tau - 1.0) * diff, tau * diff)
    return loss.mean()