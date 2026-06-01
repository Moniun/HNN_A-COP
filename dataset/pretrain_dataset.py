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
            
        # 5. 利用官方单图仿真器接口，在时间轴上执行逐帧物理差分
        td_list, sd_list = [], []
        
        for t in range(total_steps):
            img_target = cop_frames_list[t]
            img_ref = cop_frames_list[t - 1] if t > 0 else None
            
            _, _, td_tensor, sd0_tensor, sd1_tensor = run_sim_singleimg(
                img_target=img_target,
                img_ref=img_ref,
                sensor_width=640,
                sensor_height=320,
                xy=False,       
                interp=False,   
                device=torch.device('cpu')
            )
            
            # 🚀【终极维度防御隔离】：彻底不管仿真器里面有几维，直接一刀切 reshape 成标准形状
            # 强制将时间差分（TD）抽离出 160x160 纯 2D 平面，规整为单个通道 [1, 160, 160]
            td_2d = td_tensor.view(-1, 160, 160)[0] # 稳健提取首层单张二维图
            td_combined = td_2d.unsqueeze(0)        # 强重塑为 [1, 160, 160]
            
            # 强制将空间差分（SD0, SD1）规整为独立的 160x160 二维平面
            sd0_2d = sd0_tensor.view(-1, 160, 160)[0]
            sd1_2d = sd1_tensor.view(-1, 160, 160)[0]
            
            # 沿通道轴（dim=0）将其拼接为标准的两通道空间差分图 [2, 160, 160]
            sd_combined = torch.cat([sd0_2d.unsqueeze(0), sd1_2d.unsqueeze(0)], dim=0) # 死死卡在 [2, 160, 160]
            
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