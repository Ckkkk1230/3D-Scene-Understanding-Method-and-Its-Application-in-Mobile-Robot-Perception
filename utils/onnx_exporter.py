import torch
import os
from utils.logger import get_logger

logger = get_logger()

def export_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_shape: tuple = (1, 4, 256, 256),  
    opset_version: int = 12,
    dynamic_axes: bool = True
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()
    dummy_input = torch.randn(input_shape).cuda() if torch.cuda.is_available() else torch.randn(input_shape)
    d_axes = None
    if dynamic_axes:
        d_axes = {
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'semantic': {0: 'batch_size', 2: 'height', 3: 'width'},
            'depth': {0: 'batch_size', 2: 'height', 3: 'width'},
            'obstacle': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['semantic', 'depth', 'obstacle'],
        dynamic_axes=d_axes,
        verbose=False
    )
    logger.info(f"ONNX模型导出完成：{output_path}，输入形状：{input_shape}")

def check_onnx_model(path: str) -> bool:
    try:
        import onnx
        model = onnx.load(path)
        onnx.checker.check_model(model)
        logger.info("ONNX模型校验通过")
        return True
    except Exception as e:
        logger.error(f"ONNX模型校验失败：{e}")
        return False