import argparse
import os
from typing import Optional
from utils.config import load_config
from utils.logger import get_logger
from utils.checkpoint import ensure_dirs
from models.trainer import Trainer
from models.inference import run_image_inference, run_camera_inference, run_onnx_export

logger = get_logger()

def cmd_train(cfg):
    ensure_dirs(cfg)
    logger.info("===== 开始多任务联合训练（语义/深度/障碍物） =====")
    trainer = Trainer(cfg)
    trainer.train()
    logger.info("===== 训练+轻量化完成 =====")

def cmd_image_infer(cfg, input_path: Optional[str], input_dir: Optional[str]):
    ensure_dirs(cfg)
    logger.info("===== 开始图像多任务推理 =====")
    run_image_inference(cfg, input_path=input_path, input_dir=input_dir)
    logger.info("===== 图像推理完成 =====")

def cmd_camera_infer(cfg, cam_id: int, save_frames: bool):
    ensure_dirs(cfg)
    logger.info("===== 开始相机实时3D场景感知 =====")
    run_camera_inference(cfg, cam_id=cam_id, save_frames=save_frames)
    logger.info("===== 相机实时感知结束 =====")

def cmd_export_onnx(cfg):
    ensure_dirs(cfg)
    logger.info("===== 开始导出ONNX模型 =====")
    run_onnx_export(cfg)
    logger.info("===== ONNX模型导出完成 =====")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="===== 深度引导的轻量级3D场景理解-机器人感知框架 =====")
    parser.add_argument("action", choices=["train", "image_infer", "camera_infer", "export_onnx"],
                        help="执行动作：train(训练) | image_infer(图像推理) | camera_infer(相机实时推理) | export_onnx(导出ONNX)")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径，默认：config.yaml")
    parser.add_argument("--ckpt_path", default=None, help="模型权重路径，默认使用最新权重")
    parser.add_argument("--input_path", default=None, help="单张图像推理路径（仅image_infer）")
    parser.add_argument("--input_dir", default=None, help="批量图像推理目录（仅image_infer）")
    parser.add_argument("--cam_id", type=int, default=0, help="相机ID（仅camera_infer），默认：0（USB相机）")
    parser.add_argument("--save_frames", action="store_true", help="是否保存相机帧（仅camera_infer）")
    return parser.parse_args()

def main():
    args = parse_args()
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"配置文件不存在：{args.config}")
    cfg = load_config(args.config)
    logger.info(f"加载配置文件：{args.config}")

    if args.action == "train":
        cmd_train(cfg)
    elif args.action == "image_infer":
        cmd_image_infer(cfg, args.input_path, args.input_dir)
    elif args.action == "camera_infer":
        cmd_camera_infer(cfg, args.cam_id, args.save_frames)
    elif args.action == "export_onnx":
        cmd_export_onnx(cfg)

if __name__ == "__main__":
    main()