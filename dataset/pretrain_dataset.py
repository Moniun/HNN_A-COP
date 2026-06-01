import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from tianmoucv.sim import run_sim_singleimg 


class TianmoucPretrainDataset(Dataset):
    """
    随机多阶段超长流式仿真数据集 (完美契合官方接口版)
    """
    def __init__(self, image_paths, base_T=10):
        super().__init__()
        self.image_paths = image_paths
        self.base_T = base_T

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        img_bgr = cv2.imread(self.image_paths[item])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H_raw, W_raw, _ = img_rgb.shape
        
        # 1. 动态生成随机的总步长
        total_steps = np.random.randint(3 * self.base_T, 6 * self.base_T + 1)
        
        # 2. 预先构建多阶段随机运动航迹
        stage_change_points = []
        curr_step_accum = 0
        while curr_step_accum < total_steps:
            segment_len = np.random.randint(8, 15)
            curr_step_accum += segment_len
            stage_change_points.append(curr_step_accum)
            
        velocities = [(np.random.randint(-5, 6), np.random.randint(-2, 3)) for _ in range(len(stage_change_points))]
        
        # 3. 安全余量自动解算
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
            
        padding_x = max_x_drift + 10
        padding_y = max_y_drift + 10
        start_x = np.random.randint(padding_x, max(padding_x + 1, W_raw - 640 - padding_x))
        start_y = np.random.randint(padding_y, max(padding_y + 1, H_raw - 320 - padding_y))
        
        # 4. 执行多阶段滑窗裁剪 (保留 0~255 原始大图形态供仿真器计算)
        cop_frames_list = []
        curr_x, curr_y = start_x, start_y
        stage_idx = 0
        
        for t in range(total_steps):
            if t >= stage_change_points[stage_idx]:
                stage_idx += 1
            vx, vy = velocities[stage_idx]
            curr_x += vx
            curr_y += vy
            
            curr_x = max(0, min(curr_x, W_raw - 640))
            curr_y = max(0, min(curr_y, H_raw - 320))
            
            crop = img_rgb[int(curr_y):int(curr_y)+320, int(curr_x):int(curr_x)+640, :] # 形状: [320, 640, 3]
            cop_frames_list.append(crop)
            
        # 5. 🚀 核心纠正：利用官方单图仿真器接口，在时间轴上执行逐帧物理差分
        td_list, sd_list = [], []
        
        for t in range(total_steps):
            # 第一帧缺少前置参考帧，我们按官方逻辑将 img_ref 设为 None 进行自适应生成
            img_target = cop_frames_list[t]
            img_ref = cop_frames_list[t - 1] if t > 0 else None
            
            # 🚀 完美对齐官方接口参数：输入 0~255 原生 H,W,C 矩阵，设定传感器物理尺寸
            # 为了契合你网络层要求的 SNN 原生 160x160 分辨率，我们将 interp 设为 False 以导出原始尺寸
            _, _, td_tensor, sd0_tensor, sd1_tensor = run_sim_singleimg(
                img_target=img_target,
                img_ref=img_ref,
                sensor_width=640,
                sensor_height=320,
                xy=False,       # 触发标准的 SDL / SDR 空间差分通路
                interp=False,   # 🚀 核心：不进行上采样，直接吐出硬件原生的 160x160 矩阵维度！
                device=torch.device('cpu')
            )
            
            # td_tensor 的原生形状是 [2, 160, 160]，按照你网络层对 AOP_TD 单通道 (1通道) 的定义：
            # 结合你之前代码中的 aop_td[0:1, ...] 逻辑，我们对其做正负脉冲相减合成单通道，或者取其第0通道：
            td_combined = (td_tensor[0:1, :, :] - td_tensor[1:2, :, :]) # [1, 160, 160]
            
            # sd 通路包含两个方向的空间差分分支，我们将其沿通道拼接为 [2, 160, 160] 格式
            sd_combined = torch.cat([sd0_tensor.unsqueeze(0), sd1_tensor.unsqueeze(0)], dim=0) # [2, 160, 160]
            
            td_list.append(td_combined)
            sd_list.append(sd_combined)
            
        # 6. 将各时步数据沿最后一个维度（Time 维）打包，无缝对接后续的 HNN 网络结构
        # 转换为：td -> [1, 160, 160, T_total], sd -> [2, 160, 160, T_total]
        final_td = torch.stack(td_list, dim=-1)
        final_sd = torch.stack(sd_list, dim=-1)
        
        # 7. 将原始 0~255 的 COP 图像流列表转换为网络期望的标准化 [3, 320, 640, T_total] 浮点张量
        cop_np_seq = np.stack(cop_frames_list, axis=0) # [T, 320, 640, 3]
        cop_np_seq_chwt = cop_np_seq.transpose(3, 1, 2, 0) # [3, 320, 640, T]
        cop_seq_tensor = torch.from_numpy(cop_np_seq_chwt).float() / 255.0
        
        return cop_seq_tensor, final_td, final_sd