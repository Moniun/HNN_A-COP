import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from tianmoucv.simulator import Simulator


class TianmoucPretrainDataset(Dataset):
    """
    随机多阶段超长流式仿真数据集
    """
    def __init__(self, image_paths, base_T=10):
        """
        参数:
            image_paths (list): 基础高分辨率大图路径列表
            base_T (int): 相机低频快门的触发基准步长 (用来控制后面大图刷新的周期间隔)
        """
        self.image_paths = image_paths
        self.base_T = base_T
        self.sim = Simulator(thresh_td=0.05, thresh_sd=0.05)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        img_bgr = cv2.imread(self.image_paths[item])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H_raw, W_raw, _ = img_rgb.shape
        
        # 🚀 1. 动态生成随机的总步长：比如在 4*base_T 到 8*base_T 之间（40 ~ 80 步）
        # 这保证了每一次 Batch 送进来的序列长度都有所变化
        total_steps = np.random.randint(3 * self.base_T, 6 * self.base_T + 1)
        
        # 🚀 2. 预先构建“多阶段复杂随机运动航迹”
        # 我们让每隔一个变化的随机步长（8~14步），滑窗就变一次向
        stage_change_points = []
        curr_step_accum = 0
        while curr_step_accum < total_steps:
            segment_len = np.random.randint(8, 15)
            curr_step_accum += segment_len
            stage_change_points.append(curr_step_accum)
            
        # 为每个运动阶段随机生成各向异性的平移速度矩阵
        # vx 在 [-5, 5] 之间, vy 在 [-2, 2] 之间随机跳动
        velocities = [(np.random.randint(-5, 6), np.random.randint(-2, 3)) for _ in range(len(stage_change_points))]
        
        # 🚀 3. 安全余量自动解算（防止超长序列滑出大图画布边界）
        # 计算整条随机航迹可能造成的最大空域累计位移
        max_x_drift, max_y_drift = 0, 0
        temp_x, temp_y = 0, 0
        stage_idx = 0
        for t in range(total_steps):
            if t >= stage_change_points[stage_idx]:
                stage_idx += 1
            vx, vy = velocities[stage_idx]
            temp_x += vx
            temp_y += vy
            max_x_drift = max(max_x_drift, abs(temp_x))
            max_y_drift = max(max_y_drift, abs(temp_y))
            
        # 安全锚定中心裁剪起点
        padding_x = max_x_drift + 10
        padding_y = max_y_drift + 10
        start_x = np.random.randint(padding_x, max(padding_x + 1, W_raw - 640 - padding_x))
        start_y = np.random.randint(padding_y, max(padding_y + 1, H_raw - 320 - padding_y))
        
        # 🚀 4. 执行多阶段流式图像裁剪
        cop_list = []
        curr_x, curr_y = start_x, start_y
        stage_idx = 0
        
        for t in range(total_steps):
            if t >= stage_change_points[stage_idx]:
                stage_idx += 1
            vx, vy = velocities[stage_idx]
            
            curr_x += vx
            curr_y += vy
            
            # 硬件视场严密保护
            curr_x = max(0, min(curr_x, W_raw - 640))
            curr_y = max(0, min(curr_y, H_raw - 320))
            
            crop = img_rgb[curr_y:curr_y+320, curr_x:curr_x+640, :]
            crop_tensor = crop.transpose(2, 0, 1).astype(np.float32) / 255.0
            cop_list.append(crop_tensor)
            
        cop_seq = np.stack(cop_list, axis=-1)  # 成功打包为 [3, 320, 640, T_total] 变长矩阵
        
        # 5. 仿真器压制：无论 T_total 是多少，simulator 都可以流式处理并吐出对应的 [160, 160, T_total] 脉冲
        aop_td, aop_sd = self.sim.generate_aop(cop_seq)
        
        return torch.from_numpy(cop_seq).float(), torch.from_numpy(aop_td).float(), torch.from_numpy(aop_sd).float()