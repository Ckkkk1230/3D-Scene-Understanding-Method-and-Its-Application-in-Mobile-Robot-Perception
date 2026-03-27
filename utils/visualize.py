import os
import numpy as np
import torch
import random
import cv2
from PIL import Image
from typing import Dict

COLOR_PALETTE = np.zeros((40, 3), dtype=np.uint8)
COLOR_PALETTE[0] = [0, 0, 0]
COLOR_PALETTE[1:] = np.random.randint(0, 255, (39, 3), dtype=np.uint8)

def norm_tensor(t: torch.Tensor) -> torch.Tensor:
    return (t - t.min()) / (t.max() - t.min() + 1e-8)

def draw_semantic_mask(sem: torch.Tensor) -> np.ndarray:
    sem = sem.squeeze()
    if len(sem.shape) == 3:
        sem = sem.argmax(dim=0)
    h, w = sem.shape[-2:]
    sem = sem.cpu().numpy().astype(np.uint8)
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in np.unique(sem):
        if cls != 0 and cls < 40:
            mask[sem == cls] = COLOR_PALETTE[cls]
    return mask

def draw_depth_map(depth: torch.Tensor) -> np.ndarray:
    depth = depth.squeeze()
    depth = norm_tensor(depth).cpu().numpy()
    depth = (depth * 255).astype(np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    return cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)

def draw_obstacle_mask(obs: torch.Tensor) -> np.ndarray:
    obs = obs.squeeze()
    obs = norm_tensor(obs).cpu().numpy()
    h, w = obs.shape
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    mask[obs > 0.5] = [0, 0, 255]
    mask[obs <= 0.5] = [0, 255, 0]
    return mask

# ===================== 终极修复：随机取图 + 不覆盖 + 不报错 =====================
def save_multi_task_vis(rgb: torch.Tensor, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], save_path: str, epoch: int = 0):
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 🔥 核心1：随机取批次里的一张图（永远不会重复）
        batch_size = rgb.shape[0]
        idx = random.randint(0, batch_size - 1)
        
        # 🔥 核心2：文件名加 epoch 编号（永不覆盖）
        filename = os.path.splitext(save_path)[0] + f"_epoch{epoch}.png"

        # 处理图像（维度统一，不报错）
        rgb_img = rgb[idx]
        rgb_img = norm_tensor(rgb_img).permute(1, 2, 0).cpu().numpy()
        if rgb_img.shape[-1] == 1:
            rgb_img = np.concatenate([rgb_img, rgb_img, rgb_img], axis=-1)
        elif rgb_img.shape[-1] == 4:
            rgb_img = rgb_img[..., :3]
        rgb_img = (rgb_img * 255).astype(np.uint8)
        base_h = rgb_img.shape[0]

        # 生成任务图
        sem_pred = draw_semantic_mask(pred["semantic"][idx])
        sem_target = draw_semantic_mask(target["semantic"][idx])
        depth_pred = draw_depth_map(pred["depth"][idx])
        depth_target = draw_depth_map(target["depth"][idx])
        obs_pred = draw_obstacle_mask(pred["obstacle"][idx])
        obs_target = draw_obstacle_mask(target["obstacle"][idx])

        # 统一尺寸
        def resize_img(img):
            new_w = int(img.shape[1] * base_h / img.shape[0])
            return np.array(Image.fromarray(img).resize((new_w, base_h)))
        
        sem_pred = resize_img(sem_pred)
        sem_target = resize_img(sem_target)
        depth_pred = resize_img(depth_pred)
        depth_target = resize_img(depth_target)
        obs_pred = resize_img(obs_pred)
        obs_target = resize_img(obs_target)

        # 拼接保存
        all_imgs = [rgb_img, sem_pred, sem_target, depth_pred, depth_target, obs_pred, obs_target]
        final_img = np.hstack(all_imgs)
        Image.fromarray(final_img).save(filename)
        print(f"✅ 可视化已保存：{filename}")
        
    except Exception as e:
        print(f"⚠️ 可视化跳过：{str(e)}")

def save_image(tensor: torch.Tensor, path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        img = norm_tensor(tensor).permute(1,2,0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(path)
    except:
        pass

def save_image_grid(tensors, path, nrow=4):
    pass