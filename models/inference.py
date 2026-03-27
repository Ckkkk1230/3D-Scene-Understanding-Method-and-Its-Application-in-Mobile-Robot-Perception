from typing import Optional, Dict
import os
import time
import torch
import cv2
import glob
from PIL import Image
from utils.logger import get_logger
from utils.checkpoint import load_checkpoint
from utils.visualize import (
    save_multi_task_vis, draw_semantic_mask, 
    draw_depth_map, draw_obstacle_mask, norm_tensor
)
from utils.metrics import compute_fps
from utils.onnx_exporter import export_onnx, check_onnx_model
from data.camera_capture import CameraCapture
from data.dataset import SceneDataset
from models.generator import MultiTaskUNet


logger = get_logger()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"推理设备：{DEVICE}")

def _load_model(cfg: Dict, ckpt_path: Optional[str] = None) -> MultiTaskUNet:

    model = MultiTaskUNet(
        in_channels=cfg["model"]["in_channels"],
        base_channels=cfg["model"]["base_channels"],
        semantic_out=cfg["model"]["semantic_out"],
        depth_out=cfg["model"]["depth_out"],
        obstacle_out=cfg["model"]["obstacle_out"]
    ).to(DEVICE)

    if ckpt_path and os.path.isfile(ckpt_path):
        state = load_checkpoint(ckpt_path)

        model.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state)
        logger.info(f"成功加载模型权重：{ckpt_path}")
    model.eval()
    return model

def _get_latest_ckpt(ckpt_dir: str) -> str:

    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, 'epoch_*.pt')))
    if not ckpts:
        raise FileNotFoundError("未找到模型权重文件，请先训练模型")
    return ckpts[-1]

@torch.no_grad()
def run_image_inference(
    cfg: Dict,
    input_path: Optional[str] = None,
    input_dir: Optional[str] = None,
    ckpt_path: Optional[str] = None
):

    ckpt_dir = cfg["paths"]["checkpoints"]
    out_dir = os.path.join(cfg["paths"]["outputs"], "image_infer")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = ckpt_path or _get_latest_ckpt(ckpt_dir)
    model = _load_model(cfg, ckpt_path)
    image_size = cfg["dataset"]["image_size"]

    def infer_single(img_path: str):
        logger.info(f"推理图像：{img_path}")
        rgb_dir = cfg["paths"].get("nyu_images")
        depth_dir = cfg["paths"].get("nyu_depths")
        label_dir = cfg["paths"].get("nyu_labels")
        ds = SceneDataset(
            data_root=os.path.dirname(img_path),
            rgb_dir=rgb_dir,          # 传递RGB路径
            depth_dir=depth_dir,      # 传递深度路径
            label_dir=label_dir,      # 传递标签路径
            dataset_type=cfg["dataset"]["dataset_type"],
            is_train=False,
            image_size=image_size,
            num_classes=cfg["dataset"]["num_classes"],
            depth_mode=cfg["dataset"]["depth_mode"]
        )

        img_pil = Image.open(img_path).convert('RGB')
        depth_pil = ds._load_depth(img_path)
        rgb_t = ds.rgb_norm(ds.to_tensor(img_pil)).unsqueeze(0).to(DEVICE)
        depth_t = ds.depth_norm(ds.to_tensor(depth_pil)).unsqueeze(0).to(DEVICE)
        input_t = torch.cat([rgb_t, depth_t], dim=1)


        start_time = time.time()
        pred = model(input_t)
        fps = compute_fps(start_time, time.time())
        logger.info(f"单图推理速度：{fps:.2f} FPS")


        target = {
            "semantic": torch.zeros_like(pred["semantic"][0].argmax(dim=0)).unsqueeze(0).to(DEVICE),
            "depth": depth_t,
            "obstacle": torch.zeros_like(pred["obstacle"]).to(DEVICE)
        }

        img_name = os.path.basename(img_path).split('.')[0]
        vis_path = os.path.join(out_dir, f"{img_name}_multi_task_vis.png")
        save_multi_task_vis(input_t, pred, target, vis_path)
        logger.info(f"可视化结果保存：{vis_path}")

        sem_mask = draw_semantic_mask(pred["semantic"][0])
        Image.fromarray(sem_mask).save(os.path.join(out_dir, f"{img_name}_semantic.png"))

        depth_map = draw_depth_map(pred["depth"][0])
        Image.fromarray(cv2.cvtColor(depth_map, cv2.COLOR_BGR2RGB)).save(os.path.join(out_dir, f"{img_name}_depth.png"))

        obs_mask = draw_obstacle_mask(pred["obstacle"][0])
        Image.fromarray(obs_mask).save(os.path.join(out_dir, f"{img_name}_obstacle.png"))


    if input_path and os.path.isfile(input_path):
        infer_single(input_path)
    elif input_dir and os.path.isdir(input_dir):
        img_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        for f in img_files:
            infer_single(os.path.join(input_dir, f))
        logger.info(f"批量推理完成，共处理{len(img_files)}张图像，结果保存至：{out_dir}")
    else:
        raise ValueError("请指定有效的单张图像路径(--input_path)或图像目录(--input_dir)")

@torch.no_grad()
def run_camera_inference(
    cfg: Dict,
    cam_id: int = 0,
    save_frames: bool = False,
    ckpt_path: Optional[str] = None
):

    ckpt_dir = cfg["paths"]["checkpoints"]
    out_dir = os.path.join(cfg["paths"]["outputs"], "camera_frames")
    os.makedirs(out_dir, exist_ok=True) if save_frames else None
    ckpt_path = ckpt_path or _get_latest_ckpt(ckpt_dir)
    model = _load_model(cfg, ckpt_path)
    image_size = cfg["dataset"]["image_size"]

    try:
        cam = CameraCapture(cam_id=cam_id, image_size=image_size)
        logger.info(f"成功打开相机{cam_id}，开始实时推理（按q退出）")
    except Exception as e:
        logger.error(f"相机打开失败：{e}")
        return

    frame_count = 0
    total_fps = 0.0
    while True:
        try:
            raw_frame, input_t = cam.read_frame()
            input_t = input_t.to(DEVICE)

            start_time = time.time()
            pred = model(input_t)
            fps = compute_fps(start_time, time.time())
            total_fps += fps
            frame_count += 1
            avg_fps = total_fps / frame_count

            raw_frame = cv2.resize(raw_frame, (image_size, image_size))

            sem_mask = draw_semantic_mask(pred["semantic"][0])
            sem_mask = cv2.cvtColor(sem_mask, cv2.COLOR_RGB2BGR)
            raw_frame = cv2.addWeighted(raw_frame, 0.7, sem_mask, 0.3, 0)

            obs_np = norm_tensor(pred["obstacle"][0]).squeeze(0).cpu().numpy()
            contours, _ = cv2.findContours((obs_np>0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 50:
                    M = cv2.moments(cnt)
                    if M['m00'] > 0:
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        cv2.circle(raw_frame, (cx, cy), 3, (0,0,255), -1)

            cv2.putText(raw_frame, f"Avg FPS: {avg_fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(raw_frame, f"Task: Sem/Depth/Obst", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)


            cv2.imshow("Robot 3D Scene Perception (RGB+Depth)", raw_frame)
            if save_frames:
                cv2.imwrite(os.path.join(out_dir, f"frame_{frame_count:06d}.png"), raw_frame)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            logger.warning(f"帧处理失败：{e}")
            continue

    cam.release()
    cv2.destroyAllWindows()
    logger.info(f"实时推理结束，共处理{frame_count}帧，平均FPS：{avg_fps:.2f}")
    if avg_fps >= cfg["deploy"]["fps_target"]:
        logger.info(f"推理速度达标！满足PPT要求的≥{cfg['deploy']['fps_target']} FPS")
    else:
        logger.warning(f"推理速度未达标，当前{avg_fps:.2f} FPS，建议开启模型轻量化")

def run_onnx_export(cfg: Dict, ckpt_path: Optional[str] = None):

    ckpt_dir = cfg["paths"]["checkpoints"]
    onnx_out_dir = cfg["paths"]["onnx"]
    os.makedirs(onnx_out_dir, exist_ok=True)
    ckpt_path = ckpt_path or _get_latest_ckpt(ckpt_dir)
    onnx_path = os.path.join(onnx_out_dir, "multi_task_unet_robot.onnx")
    input_shape = cfg["deploy"]["input_shape"]

    model = _load_model(cfg, ckpt_path)
    export_onnx(
        model=model,
        output_path=onnx_path,
        input_shape=input_shape,
        opset_version=cfg["deploy"]["onnx_opset"],
        dynamic_axes=True
    )
    check_onnx_model(onnx_path)
    logger.info(f"ONNX模型导出+校验完成，保存至：{onnx_path}")
    logger.info(f"可通过TensorRT转换为Jetson NX最优推理格式：trtexec --onnx={onnx_path} --saveEngine=robot_model.engine")