from typing import Dict
import yaml


def load_config(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg