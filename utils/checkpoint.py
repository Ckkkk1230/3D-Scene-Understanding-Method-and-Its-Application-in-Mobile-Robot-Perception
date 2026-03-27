import os
from typing import Dict
import torch
from utils.logger import get_logger 

def save_checkpoint(state: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path: str) -> Dict:
    return torch.load(path, map_location='cpu')

def ensure_dirs(cfg: Dict) -> None:
    for k, v in cfg["paths"].items():
        os.makedirs(v, exist_ok=True)
    logger = get_logger()
    logger.info("所有目录已创建/验证完成")

