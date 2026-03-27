import cv2
import torch
import numpy as np
from PIL import Image
from utils.depth import compute_depth
from torchvision import transforms

class CameraCapture:
    def __init__(self, cam_id: int = 0, image_size: int = 256):
        self.cam_id = cam_id
        self.image_size = image_size
        self.cap = cv2.VideoCapture(cam_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开相机：{cam_id}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_size)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.depth_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def read_frame(self) -> tuple:
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("相机读取帧失败")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        
        # 修复：先保存临时文件，再用compute_depth读取（兼容原函数逻辑）
        temp_path = "temp_camera_frame.png"
        img_pil.save(temp_path)
        depth_pil = compute_depth(temp_path, mode='midas')  # 传文件路径
        import os
        os.remove(temp_path)  # 删除临时文件
        
        rgb_t = self.transform(img_pil).unsqueeze(0)  
        depth_t = self.depth_transform(depth_pil).unsqueeze(0)  
        input_t = torch.cat([rgb_t, depth_t], dim=1)  
        return frame, input_t
    
    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

def show_camera_feed(cam: CameraCapture):
    while True:
        frame, _ = cam.read_frame()
        cv2.imshow("Robot Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()