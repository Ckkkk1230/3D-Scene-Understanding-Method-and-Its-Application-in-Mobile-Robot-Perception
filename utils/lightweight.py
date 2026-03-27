import torch
import torch.nn as nn
import torch.quantization
from torch.nn.utils import prune

def prune_model(model: nn.Module, prune_ratio: float = 0.2) -> nn.Module:
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            prune.l1_unstructured(module, name='weight', amount=prune_ratio)
            prune.remove(module, 'weight')
    return model

def quantize_model(model: nn.Module, calib_data: torch.Tensor) -> nn.Module:
    model.eval()
    model = torch.quantization.quantize_dynamic(
        model, {nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d}, dtype=torch.qint8
    )

    with torch.no_grad():
        model(calib_data)
    return model

def model_lightweight(model: nn.Module, calib_data: torch.Tensor, prune_ratio: float = 0.2) -> nn.Module:
    print(f"开始模型轻量化：剪枝率{prune_ratio} + INT8量化")
    model = prune_model(model, prune_ratio)
    model = quantize_model(model, calib_data)
    print("模型轻量化完成")
    return model