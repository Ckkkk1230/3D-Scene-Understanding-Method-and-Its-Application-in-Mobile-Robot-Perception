import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def compute_rmse(pred: torch.Tensor, target: torch.Tensor, mask=None) -> float:
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    return torch.sqrt(torch.nn.functional.mse_loss(pred, target)).item()

def compute_mae(pred: torch.Tensor, target: torch.Tensor, mask=None) -> float:
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    return torch.nn.functional.l1_loss(pred, target).item()

def compute_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    pred = pred.argmax(dim=1).flatten()
    target = target.flatten()
    intersection = torch.logical_and(pred == target, target < num_classes).sum()
    union = torch.logical_or(pred == target, target < num_classes).sum()
    if union == 0:
        return 0.0
    return (intersection / union).item()

def compute_cls_metrics(pred: torch.Tensor, target: torch.Tensor) -> tuple:
    pred = pred.sigmoid().flatten() > 0.5
    target = target.flatten() > 0.5
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    p = precision_score(target_np, pred_np, zero_division=0)
    r = recall_score(target_np, pred_np, zero_division=0)
    f = f1_score(target_np, pred_np, zero_division=0)
    return p, r, f

def compute_dce(pred_depth: torch.Tensor, target_depth: torch.Tensor, mask=None) -> float:
    if mask is not None:
        pred_depth = pred_depth[mask]
        target_depth = target_depth[mask]
    pred_mean = pred_depth.mean()
    target_mean = target_depth.mean()
    cov = ((pred_depth - pred_mean) * (target_depth - target_mean)).mean()
    pred_std = pred_depth.std()
    target_std = target_depth.std()
    corr = cov / (pred_std * target_std + 1e-8)
    std_ratio = 2 * pred_std * target_std / (pred_std**2 + target_std**2 + 1e-8)
    dce = 1 - ( (1 + corr) / 2 ) * std_ratio
    return dce.item()

def compute_fps(start_time: float, end_time: float, batch_size: int = 1) -> float:
    return batch_size / (end_time - start_time + 1e-8)

class MetricMonitor:
    def __init__(self):
        self.metrics = {
            'miou': [], 'rmse': [], 'mae': [], 'precision': [],
            'recall': [], 'f1': [], 'dce': [], 'loss': []
        }
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.metrics:
                self.metrics[k].append(v)
    
    def get_avg(self) -> dict:
        return {k: np.mean(v) if v else 0.0 for k, v in self.metrics.items()}
    
    def reset(self):
        for k in self.metrics:
            self.metrics[k] = []