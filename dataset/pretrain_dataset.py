# 新建：pretrain_dataset.py
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

# 💡 导入天眸芯官方仿真器核心组件（依据官方标准库接口设计）
# 如果你的环境里仿真器接口命名有微调，可对应修改此行的导入
from tianmoucv.simulator import Simulator 


class TianmoucPretrainDataset(Dataset):
    """
    自监督预训练专用的滑窗仿真数据集
    输入一堆大于 320x640 的通用图像，通过滑动窗口模拟相机平移，生成全时步 COP 序列与对应的 AOP 脉冲
    """
    def __init__(self, image_paths, T_steps=10, velocity=(4, 2)):
        """
        参数:
            image_paths (list): 大图的路径列表 (每张图尺寸需大于 320x640)
            T_steps (int): 模拟的时间微步数 (即序列长度 T)
            velocity (tuple): 裁窗每一步在 (x, y) 方向上的平移像素速度
        """
        self.image_paths = image_paths
        self.T_steps = T_steps
        self.vx, self.vy = velocity
        
        # 初始化天眸芯官方硬件行为仿真器
        # 仿真器内部会自动把输入的 320x640 视场挤压并计算出原生的 160x160 脉冲流
        self.sim = Simulator(thresh_td=0.05, thresh_sd=0.05)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        # 1. 读取单张高分辨率大图
        img_bgr = cv2.imread(self.image_paths[item])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H_raw, W_raw, _ = img_rgb.shape
        
        # 初始裁窗左上角坐标 (设在中央区域防止越界)
        start_x = max(0, (W_raw - 640) // 2)
        start_y = max(0, (H_raw - 320) // 2)
        
        cop_list = []
        
        # 2. 模拟滑窗移动，构建包含时间轴 T 的全量 COP 图像流
        for t in range(self.T_steps):
            curr_x = start_x + t * self.vx
            curr_y = start_y + t * self.vy
            
            # 边界防御保障
            curr_x = min(curr_x, W_raw - 640)
            curr_y = min(curr_y, H_raw - 320)
            
            # 裁剪出标准的双目拼接视场尺寸: [320, 640, 3]
            crop = img_rgb[curr_y:curr_y+320, curr_x:curr_x+640, :]
            
            # 变换为 PyTorch 标准通道轴: [3, 320, 640]
            crop_tensor = crop.transpose(2, 0, 1).astype(np.float32) / 255.0
            cop_list.append(crop_tensor)
            
        # 打包成具有时间轴的 COP 全量序列: [3, 320, 640, T]
        cop_seq = np.stack(cop_list, axis=-1)
        
        # 3. 🚀 将全量图像流喂给 tianmouc 仿真器，生成对应的动作通路物理原语
        # 仿真器在内部会将 320x640 画面自适应处理，并吐出原生的 [160, 160, T] 正方形脉冲
        # aop_td 形状: [1, 160, 160, T], aop_sd 形状: [2, 160, 160, T]
        aop_td, aop_sd = self.sim.generate_aop(cop_seq) 
        
        # 转为标准的 float32 张量返回
        return (
            torch.from_numpy(cop_seq).float(),    # 全时步图像流真值 [3, 320, 640, T]
            torch.from_numpy(aop_td).float(),     # 仿真 AOP 时间差分 [1, 160, 160, T]
            torch.from_numpy(aop_sd).float()      # 仿真 AOP 空间差分 [2, 160, 160, T]
        )