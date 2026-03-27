from typing import Tuple, Optional
import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from utils.depth import compute_depth

class SceneDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        dataset_type: str = "nyu",
        is_train: bool = True,
        image_size: int = 256,
        num_classes: int = 40,
        random_crop: bool = True,
        random_flip: bool = True,
        color_jitter: bool = False,
        depth_mode: str = "midas",
    ) -> None:
        # 强制转换为绝对路径
        self.data_root = os.path.abspath(data_root)
        print(f"[数据集加载] 根目录：{self.data_root}")

        self.is_train = is_train
        self.image_size = image_size
        self.num_classes = num_classes
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.color_jitter = color_jitter
        self.depth_mode = depth_mode.lower()

        # 100%匹配你的三个文件夹
        self.img_dir = os.path.join(self.data_root, "nyu_images")
        self.dep_dir = os.path.join(self.data_root, "nyu_depths")
        self.lbl_dir = os.path.join(self.data_root, "nyu_labels")

        # 提前检查文件夹是否存在
        for dir_path, dir_name in zip(
            [self.img_dir, self.dep_dir, self.lbl_dir],
            ["nyu_images", "nyu_depths", "nyu_labels"]
        ):
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"文件夹不存在：{dir_path}，请检查路径")
            print(f"[数据集加载] 找到{dir_name}文件夹")

        # 加载文件列表，支持不同后缀自动匹配
        self._load_and_validate_files()

        # 图像预处理
        self.to_tensor = T.ToTensor()
        self.rgb_norm = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.depth_norm = T.Normalize(mean=[0.5], std=[0.5])
        
        # 兼容所有PyTorch版本的插值模式
        try:
            from torchvision.transforms import InterpolationMode
            self.resize = T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC)
            self.label_resize = T.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST)
        except ImportError:
            self.resize = T.Resize((image_size, image_size), interpolation='bicubic')
            self.label_resize = T.Resize((image_size, image_size), interpolation='nearest')
        
        self.jitter = T.ColorJitter(0.2, 0.2, 0.2, 0.05) if color_jitter else None
        print(f"[数据集加载] 完成，共{len(self)}张有效图像")

    def _find_matching_file(self, base_name: str, target_dir: str) -> Optional[str]:
        """核心功能：在目标文件夹里找同名但不同后缀的文件"""
        # 支持的所有图片后缀
        valid_exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".bmp", ".BMP")
        
        # 遍历目标文件夹，找文件名匹配的
        for f in os.listdir(target_dir):
            # 去掉后缀，比较纯文件名
            f_base = os.path.splitext(f)[0]
            if f_base == base_name and f.lower().endswith(valid_exts):
                return os.path.join(target_dir, f)
        return None

    def _load_and_validate_files(self):
        # 读取RGB图像列表
        img_ext = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
        img_files = [f for f in os.listdir(self.img_dir) if f.lower().endswith(img_ext)]
        
        if not img_files:
            raise FileNotFoundError(f"nyu_images文件夹里没有找到任何图片，请检查文件")

        # 校验每个文件的对应关系，支持不同后缀
        self.images = []
        self.semantics = []
        self.depths = []

        for img_filename in img_files:
            # 1. RGB文件路径
            img_path = os.path.join(self.img_dir, img_filename)
            # 2. 提取纯文件名（不带后缀）
            base_name = os.path.splitext(img_filename)[0]
            
            # 3. 在nyu_labels里找同名文件（不管后缀是.jpg还是.png）
            lbl_path = self._find_matching_file(base_name, self.lbl_dir)
            # 4. 在nyu_depths里找同名文件（不管后缀是.jpg还是.png）
            dep_path = self._find_matching_file(base_name, self.dep_dir)

            # 只有三个文件都找到，才加入列表
            if img_path and lbl_path and dep_path:
                self.images.append(img_path)
                self.semantics.append(lbl_path)
                self.depths.append(dep_path)
                # 打印匹配信息（可选，方便调试）
                # print(f"[匹配成功] {img_filename} -> {os.path.basename(lbl_path)} -> {os.path.basename(dep_path)}")
            else:
                print(f"[警告] 跳过文件{img_filename}：未找到对应标签/深度图")

        if len(self.images) == 0:
            raise FileNotFoundError("没有找到任何匹配的RGB、标签、深度图，请检查文件名是否一致（纯文件名必须相同，后缀可以不同）")

    def _load_rgb(self, path: str) -> Image.Image:
        return Image.open(path).convert('RGB')

    def _load_semantic(self, path: str) -> torch.Tensor:
      sem = Image.open(path).convert('L')
      sem = self.label_resize(sem)
      sem_np = np.array(sem)
      # 把所有超出0~39的像素（包括255）替换为背景类0
      sem_np[(sem_np < 0) | (sem_np >= 40)] = 0
      sem = torch.from_numpy(sem_np).long()
      return sem

    def _load_depth(self, path: str) -> Image.Image:
        if os.path.exists(path):
            return Image.open(path).convert('L')
        return compute_depth(path.replace("nyu_depths", "nyu_images"), mode=self.depth_mode)

    def _load_obstacle(self, path: str) -> torch.Tensor:
        # 从语义标签自动生成障碍物掩码，无需额外文件
        sem = Image.open(path).convert('L')
        sem = self.label_resize(sem)
        sem_np = np.array(sem)
        obs_np = (sem_np > 0).astype(np.float32)
        return torch.from_numpy(obs_np).float()

    def _augment(self, rgb: Image.Image, depth: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if self.random_flip and random.random() < 0.5:
            rgb = T.functional.hflip(rgb)
            depth = T.functional.hflip(depth)
        rgb = self.resize(rgb)
        depth = self.resize(depth)
        if self.jitter is not None:
            rgb = self.jitter(rgb)
        return rgb, depth

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        rgb = self._load_rgb(self.images[idx])
        semantic_t = self._load_semantic(self.semantics[idx])
        depth = self._load_depth(self.depths[idx])
        obstacle_t = self._load_obstacle(self.semantics[idx])

        rgb, depth = self._augment(rgb, depth)

        rgb_t = self.rgb_norm(self.to_tensor(rgb))
        depth_t = self.depth_norm(self.to_tensor(depth))
        input_t = torch.cat([rgb_t, depth_t], dim=0)

        labels = {
            "semantic": semantic_t,
            "depth": depth_t,
            "obstacle": obstacle_t
        }

        return input_t, labels