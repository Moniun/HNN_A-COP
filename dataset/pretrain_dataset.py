import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from tianmoucv.sim import run_sim_singleimg
import torch.nn.functional as F


def torch_td_denoise(td_tensor, var_fil_ksize=3, var_th=0.5, adapt_th_min=5, adapt_th_max=8):
    """
    PyTorch 版本的 TD 去噪（与官方 denoise_defualt_args 参数一致）
    
    参数:
        td_tensor: [H, W] 或 [B, H, W] 的 tensor
        var_fil_ksize: 方差滤波核大小（官方默认: 3）
        var_th: 方差阈值（官方默认: 0.5）
        adapt_th_min: 自适应阈值最小值（官方默认: 3）
        adapt_th_max: 自适应阈值最大值（官方默认: 8）
    """
    if td_tensor.dim() == 2:
        td_tensor = td_tensor.unsqueeze(0)
    
    pad = var_fil_ksize // 2
    td_padded = F.pad(td_tensor, [pad]*4, mode='reflect')
    windows = td_padded.unfold(1, var_fil_ksize, 1).unfold(2, var_fil_ksize, 1)
    windows = windows.contiguous().view(td_tensor.shape[0], td_tensor.shape[1], td_tensor.shape[2], -1)
    
    var = torch.var(windows, dim=-1)
    mask = var > var_th
    adapt_th = adapt_th_min + (adapt_th_max - adapt_th_min) * (var / var.max())
    
    td_denoised = td_tensor.clone()
    td_denoised[mask & (torch.abs(td_tensor) < adapt_th)] = 0
    return td_denoised.squeeze()

def torch_sd_denoise(sd_tensor, var_fil_ksize=3, var_th=1.0, adapt_th_min=8, adapt_th_max=15):
    """PyTorch 版本的 SD 去噪（与官方参数一致）"""
    sdl_denoised = torch_td_denoise(sd_tensor[0], var_fil_ksize, var_th, adapt_th_min, adapt_th_max)
    sdr_denoised = torch_td_denoise(sd_tensor[1], var_fil_ksize, var_th, adapt_th_min, adapt_th_max)
    return torch.stack([sdl_denoised, sdr_denoised], dim=0)

class TianmoucPretrainDataset(Dataset):
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
        
        # 大Batch对齐优化固定时步
        total_steps = 40 
        
        # 预先构建高动态运动航迹
        stage_change_points = []
        curr_step_accum = 0
        while curr_step_accum < total_steps:
            segment_len = np.random.randint(8, 15)
            curr_step_accum += segment_len
            stage_change_points.append(curr_step_accum)
            
        # 保持高动态速度运动设定（激发大核脉冲优势）
        velocities = [(np.random.randint(-20, 21), np.random.randint(-10, 11)) for _ in range(len(stage_change_points))]
        
        # 安全余量自动解算
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
            
        padding_x = max_x_drift + 15
        padding_y = max_y_drift + 15
        start_x = np.random.randint(padding_x, max(padding_x + 1, W_raw - 640 - padding_x))
        start_y = np.random.randint(padding_y, max(padding_y + 1, H_raw - 320 - padding_y))
        
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
            
            crop = img_rgb[int(curr_y):int(curr_y)+320, int(curr_x):int(curr_x)+640, :]
            cop_frames_list.append(crop)
            
        # 执行仿真差分
        td_list, sd_list = [], []
        for t in range(total_steps):
            img_target = cop_frames_list[t]
            img_ref = cop_frames_list[t - 1] if t > 0 else cop_frames_list[t]
            
            _, _, td_tensor, sd0_tensor, sd1_tensor = run_sim_singleimg(
                img_target=img_target,
                img_ref=img_ref,
                sensor_width=640,
                sensor_height=320,
                xy=False,       
                interp=False,   
                device=torch.device('cpu'),
                # ✅ 关闭所有噪声参数
                sensor_fixed_noise_prob=0.0,
                sensor_random_noise_prob=0.0,
                sensor_fixed_noise_std_ch0=0.0,
                sensor_fixed_noise_std_ch12=0.0,
                sensor_random_noise_std=0.0,
                sensor_poisson_lambda=0,
                gray_weight_jitter=0.0,
                gray_gain_min=1.0,
                gray_gain_max=1.0,
                sim_threshold_range=(0.0, 0.0),
                
                # ✅ 新增：强制关闭 FPN 噪声（通过设置参数使噪声为零）
                sensor_fixed_noise_mean_ch0=0.0,  # 均值设为0
                sensor_fixed_noise_mean_ch12=0.0   # 均值设为0
            )
            
            # 去噪
            td_tensor = torch_td_denoise(td_tensor.float())
            sd0_tensor = torch_sd_denoise(sd0_tensor.float())
            sd1_tensor = torch_sd_denoise(sd1_tensor.float())

            td_2d = td_tensor.view(-1, 160, 160)[0] 
            td_combined = td_2d.unsqueeze(0)        
            
            sd0_2d = sd0_tensor.view(-1, 160, 160)[0]
            sd1_2d = sd1_tensor.view(-1, 160, 160)[0]
            sd_combined = torch.cat([sd0_2d.unsqueeze(0), sd1_2d.unsqueeze(0)], dim=0) 
            
            td_list.append(td_combined)
            sd_list.append(sd_combined)
            
        final_td = torch.stack(td_list, dim=-1)
        final_sd = torch.stack(sd_list, dim=-1)
        
        # 🔒 交付百分之百干净高保真的原生流序列
        cop_np_seq = np.stack(cop_frames_list, axis=0) 
        cop_np_seq_chwt = cop_np_seq.transpose(3, 1, 2, 0) 
        cop_seq_tensor = torch.from_numpy(cop_np_seq_chwt).float() / 255.0
        
        return cop_seq_tensor, final_td, final_sd