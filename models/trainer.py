from typing import Dict
import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from utils.logger import get_logger
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.visualize import save_multi_task_vis
from utils.metrics import MetricMonitor, compute_miou, compute_rmse, compute_mae, compute_cls_metrics, compute_dce
from utils.lightweight import model_lightweight
from data.loader import create_dataloaders
from models.generator import MultiTaskUNet
import torch.nn.functional as F

class Trainer:
    def __init__(self, cfg: Dict) -> None:
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = get_logger()
        self.logger.info(f"训练设备：{self.device}")

        self.train_loader, self.val_loader = create_dataloaders(cfg)

        self.logger.info(f"训练集大小：{len(self.train_loader.dataset)}，验证集大小：{len(self.val_loader.dataset)}")
        self.model = MultiTaskUNet(
            in_channels=cfg["model"]["in_channels"],
            base_channels=cfg["model"]["base_channels"],
            semantic_out=cfg["model"]["semantic_out"],
            depth_out=cfg["model"]["depth_out"],
            obstacle_out=cfg["model"]["obstacle_out"]
        ).to(self.device)
        self.sem_loss = nn.CrossEntropyLoss(ignore_index=255)  
        self.depth_loss = nn.L1Loss()          
        self.obs_loss = nn.BCEWithLogitsLoss() 
        self.dce_loss = nn.L1Loss()            
        self.lr = cfg["train"]["lr"]
        self.beta1 = cfg["train"]["beta1"]
        self.beta2 = cfg["train"]["beta2"]
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2)
        )
        self.epochs = cfg["train"]["epochs"]
        self.save_every = cfg["train"]["save_every"]
        self.loss_weight = cfg["train"]["loss_weight"]
        self.checkpoint_dir = cfg["paths"]["checkpoints"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.train_metrics = MetricMonitor()
        self.val_metrics = MetricMonitor()
        self.save_dir = cfg.get("train", {}).get("save_dir", "./output")
        os.makedirs(self.save_dir, exist_ok=True)

    def _compute_multi_task_loss(self, pred: dict, target: dict) -> tuple:
    # 1. 语义分割损失：上采样到标签尺寸 + 忽略255无效值
      pred_sem_up = F.interpolate(
        pred["semantic"], 
        size=target["semantic"].shape[1:],  # 对齐到标签的 [256,256]
        mode='bilinear', 
        align_corners=False
        )
      sem_loss = self.sem_loss(pred_sem_up, target["semantic"])

      # 2. 深度估计损失（尺寸已对齐，直接计算）
      depth_loss = self.depth_loss(pred["depth"], target["depth"])

      # 3. 障碍物检测损失：给标签扩展通道维度（核心修复！）
      # 模型输出：[16, 1, 256, 256] → 标签扩展后：[16, 1, 256, 256]
      target_obstacle = target["obstacle"].unsqueeze(1)  # 新增：扩展通道维度
      obs_loss = self.obs_loss(pred["obstacle"], target_obstacle)  # 修改：用扩展后的标签

      # 4. DCE损失（修复：只传入depth分支，而非整个字典）
      dce_loss = self.dce_loss(pred["depth"], target["depth"]) if self.dce_loss else 0.0

      # 5. 加权总损失
      total_loss = (
          self.cfg["train"]["loss_weight"]["semantic"] * sem_loss +
          self.cfg["train"]["loss_weight"]["depth"] * depth_loss +
          self.cfg["train"]["loss_weight"]["obstacle"] * obs_loss +
          self.cfg["train"]["loss_weight"]["dce"] * dce_loss
      )
      return total_loss, sem_loss, depth_loss, obs_loss, dce_loss

    def _train_epoch(self, epoch: int) -> None:
        self.model.train()
        self.train_metrics.reset()
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}/{self.epochs}")
        for input_t, target in pbar:
            input_t = input_t.to(self.device)
            for k in target:
                target[k] = target[k].to(self.device)
            pred = self.model(input_t)
            total_loss, sem_loss, depth_loss, obs_loss, dce_loss = self._compute_multi_task_loss(pred, target)
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            miou = compute_miou(pred["semantic"], target["semantic"], self.cfg["dataset"]["num_classes"])
            rmse = compute_rmse(pred["depth"], target["depth"])
            mae = compute_mae(pred["depth"], target["depth"])
            p, r, f = compute_cls_metrics(pred["obstacle"], target["obstacle"])
            dce = compute_dce(pred["depth"], target["depth"])

            self.train_metrics.update(
                loss=total_loss.item(), miou=miou, rmse=rmse, mae=mae,
                precision=p, recall=r, f1=f, dce=dce
            )
            pbar.set_postfix(self.train_metrics.get_avg())

        avg_metrics = self.train_metrics.get_avg()
        self.logger.info(f"Train Epoch {epoch} | Avg Metrics: {avg_metrics}")

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> None:
        self.model.eval()
        self.val_metrics.reset()
        pbar = tqdm(self.val_loader, desc=f"Val Epoch {epoch}/{self.epochs}")
        for i, (input_t, target) in enumerate(pbar):
            input_t = input_t.to(self.device)
            for k in target:
                target[k] = target[k].to(self.device)
            pred = self.model(input_t)
            total_loss, _, _, _, _ = self._compute_multi_task_loss(pred, target)

            miou = compute_miou(pred["semantic"], target["semantic"], self.cfg["dataset"]["num_classes"])
            rmse = compute_rmse(pred["depth"], target["depth"])
            mae = compute_mae(pred["depth"], target["depth"])
            p, r, f = compute_cls_metrics(pred["obstacle"], target["obstacle"])
            dce = compute_dce(pred["depth"], target["depth"])

            self.val_metrics.update(
                loss=total_loss.item(), miou=miou, rmse=rmse, mae=mae,
                precision=p, recall=r, f1=f, dce=dce
            )

            if i == 0 and epoch % 1 == 0:
                vis_dir = os.path.join(self.save_dir, "vis", f"epoch_{epoch}")
                vis_path = os.path.join(vis_dir, "multi_vis.png")
                save_multi_task_vis(input_t, pred, target, vis_path)

            pbar.set_postfix(self.val_metrics.get_avg())

        avg_metrics = self.val_metrics.get_avg()
        self.logger.info(f"Val Epoch {epoch} | Avg Metrics: {avg_metrics}")

    def _save_checkpoint(self, epoch: int) -> None:
        ckpt_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}.pt")
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metrics': self.val_metrics.get_avg()
        }, ckpt_path)
        self.logger.info(f"模型权重保存至：{ckpt_path}")

    def train(self) -> None:
        self.logger.info("开始多任务联合训练...")
        for epoch in range(1, self.epochs + 1):
            self._train_epoch(epoch)
            self._val_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

        self.logger.info("训练完成，开始模型轻量化...")
        calib_data = torch.randn(1, self.cfg["model"]["in_channels"], self.cfg["dataset"]["image_size"], self.cfg["dataset"]["image_size"]).to(self.device)
        self.model = model_lightweight(self.model, calib_data, self.cfg["lightweight"]["prune_ratio"])
        self._save_checkpoint(self.epochs + 1)
        self.logger.info("多任务联合训练+轻量化完成！")