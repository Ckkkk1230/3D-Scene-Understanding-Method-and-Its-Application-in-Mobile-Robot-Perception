from typing import Dict, Tuple
import os
from torch.utils.data import DataLoader
from data.dataset import SceneDataset

def create_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader]:
    ds_args = cfg["dataset"]
    train_ds = SceneDataset(
        data_root=cfg["paths"]["data_root"],
        dataset_type=ds_args["dataset_type"],
        is_train=True,
        image_size=ds_args["image_size"],
        num_classes=ds_args["num_classes"],
        random_crop=ds_args["augment"]["random_crop"],
        random_flip=ds_args["augment"]["random_flip"],
        color_jitter=ds_args["augment"]["color_jitter"],
        depth_mode=ds_args["depth_mode"]
    )
    val_ds = SceneDataset(
        data_root=cfg["paths"]["data_root"],
        dataset_type=ds_args["dataset_type"],
        is_train=False,
        image_size=ds_args["image_size"],
        num_classes=ds_args["num_classes"],
        random_crop=False,
        random_flip=False,
        color_jitter=False,
        depth_mode=ds_args["depth_mode"]
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        drop_last=False
    )
    return train_loader, val_loader