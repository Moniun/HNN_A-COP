# /root/autodl-tmp/HNN_A-COP/data/mot_to_tianmouc.py
import os
import glob
import numpy as np
import cv2
import torch
from tqdm import tqdm
from tianmoucv.sim import run_sim_singleimg

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

def parse_gt_txt(gt_path):
    """高效解析 MOT17 原生真实时序标签，保留 object_id 用于死锁追踪"""
    frame_dict = {}
    if not os.path.exists(gt_path): return frame_dict
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7: continue
            frame_id = int(parts[0])
            obj_id = int(parts[1]) # 🔒 提取珍贵的真实轨迹 ID
            class_id = int(parts[7])
            if class_id not in [1, 3]: continue  # 只提纯行人和车辆
            
            x1, y1, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            # 存储格式增加包含 obj_id: [obj_id, cx, cy, w, h]
            frame_dict.setdefault(frame_id, []).append([obj_id, x1 + w/2.0, y1 + h/2.0, w, h])
    return frame_dict

def convert_mot_to_tianmouc(src_base_dir, output_dir, T_steps=40, is_test_set=False):
    os.makedirs(output_dir, exist_ok=True)
    seq_paths = sorted([d for d in glob.glob(os.path.join(src_base_dir, "*")) if os.path.isdir(d)])
    global_seq_idx = 0
    
    for seq_path in seq_paths:
        seq_name = os.path.basename(seq_path)
        img_dir, gt_path = os.path.join(seq_path, "img1"), os.path.join(seq_path, "gt", "gt.txt")
        all_imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        gt_dict = parse_gt_txt(gt_path)
        
        print(f"🎬 正在全量加载视频序列: {seq_name} (共 {len(all_imgs)} 帧)")
        full_video_cop = []
        for img_path in all_imgs:
            frame_bgr = cv2.imread(img_path)
            frame_resized = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), (640, 320))
            full_video_cop.append(frame_resized)
            
        print(f"⚡ 正在全量榨取常开高保真天眸 AOP 差分脉冲特征...")
        full_td_list, full_sd_list = [], []
        for t in range(len(all_imgs)):
            img_target = full_video_cop[t]
            img_ref = cop_frames_list[t - 1] if t > 0 else cop_frames_list[t]
            
            _, _, td_tensor, sd0_tensor, sd1_tensor = run_sim_singleimg(
                img_target=img_target, img_ref=img_ref,
                sensor_width=640, 
                sensor_height=320, 
                xy=False, 
                interp=True, 
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
            
            full_td_list.append(td_tensor.view(-1, 160, 160)[0].unsqueeze(0).numpy())
            full_sd_list.append(torch.cat([sd0_tensor.view(-1, 160, 160)[0].unsqueeze(0), sd1_tensor.view(-1, 160, 160)[0].unsqueeze(0)], dim=0).numpy())
            
        full_cop_arr = np.stack(full_video_cop, axis=0).transpose(3, 1, 2, 0)
        full_td_arr = np.stack(full_td_list, axis=-1)
        full_sd_arr = np.stack(full_sd_list, axis=-1)
        
        # 3. 🔒 核心重构：后端分块切片并执行“跨帧 ID 死锁匹配”
        num_segments = len(all_imgs) // T_steps
        for seg_idx in range(num_segments):
            s_t, e_t = seg_idx * T_steps, seg_idx * T_steps + T_steps
            
            segment_labels_list = []
            
            if is_test_set:
                # 测试集不需要真实标签，填充齐整的全零占位矩阵
                for t in range(T_steps):
                    segment_labels_list.append(np.zeros((3, 4), dtype=np.float32))
            else:
                # 🔒 黄金锁定机制：在当前 40 帧片段的第 0 帧，挑出面积最大的前 3 个目标的真实 ID
                start_real_frame_id = s_t + 1
                start_objects = gt_dict.get(start_real_frame_id, [])
                
                # 按物体面积 (w * h) 从大到小排序，锁定最显著的目标
                start_objects_sorted = sorted(start_objects, key=lambda x: x[3] * x[4], reverse=True)
                locked_ids = [obj[0] for obj in start_objects_sorted[:3]]
                
                # 如果当前片段开局没有任何车或行人，直接跳过不浪费算力
                if len(locked_ids) == 0:
                    continue
                
                # 重新去原视频里核对，读取原图尺寸以便进行坐标压缩
                sample_img = cv2.imread(all_imgs[s_t])
                H_orig, W_orig, _ = sample_img.shape
                
                # 开始在时间轴上推进 40 步
                for t in range(T_steps):
                    real_frame_id = s_t + t + 1
                    current_frame_all_objects = gt_dict.get(real_frame_id, [])
                    
                    # 建立 {obj_id: [cx, cy, w, h]} 的快速索引字典
                    current_obj_idx_dict = {obj[0]: obj[1:] for obj in current_frame_all_objects}
                    
                    formatted_labels = np.zeros((3, 4), dtype=np.float32)
                    # 🔒 死锁防线：按照第 0 帧锁定的固定 ID 顺序，去给当前的 3 个插槽填框
                    for slot_idx, target_id in enumerate(locked_ids):
                        if target_id in current_obj_idx_dict:
                            cx, cy, w, h = current_obj_idx_dict[target_id]
                            # 相对归一化压缩到 [0, 1] 空间，抹平 MSE 平方惩罚效应
                            formatted_labels[slot_idx] = [
                                np.clip(cx / W_orig, 0, 1),
                                np.clip(cy / H_orig, 0, 1),
                                w / W_orig,
                                h / H_orig
                            ]
                        else:
                            # 如果该目标中途断档或被完全遮挡，该插槽保持全零占位，绝对不填别人！
                            pass
                    segment_labels_list.append(formatted_labels)
            
            segment_labels = np.stack(segment_labels_list, axis=-1) # [3, 4, 40]
            
            # 如果这 40 帧里全是全零（目标中途全都消失了），抛弃该片段，保证数据纯净
            if not is_test_set and np.allclose(segment_labels, 0):
                continue
                
            np.save(os.path.join(output_dir, f"{seq_name}_seg_{seg_idx:02d}_cop_t0.npy"), full_cop_arr[..., s_t:e_t])
            np.save(os.path.join(output_dir, f"{seq_name}_seg_{seg_idx:02d}_aop_td.npy"), full_td_arr[..., s_t:e_t])
            np.save(os.path.join(output_dir, f"{seq_name}_seg_{seg_idx:02d}_aop_sd.npy"), full_sd_arr[..., s_t:e_t])
            np.save(os.path.join(output_dir, f"{seq_name}_seg_{seg_idx:02d}_label.npy"), segment_labels)
            global_seq_idx += 1

    print(f"\n======================= 🎉 MOT17 跨帧轨迹 ID 死锁固化成功！ =======================")

if __name__ == "__main__":
    convert_mot_to_tianmouc(
        src_base_dir="/root/autodl-tmp/HNN_A-COP/data/mot17_project/extracted_mot17/MOT17/train",
        output_dir="/root/autodl-tmp/HNN_A-COP/data/mot17_project/tianmouc_vid_proc/train",
        T_steps=40,
        is_test_set=False
    )
    convert_mot_to_tianmouc(
        src_base_dir="/root/autodl-tmp/HNN_A-COP/data/mot17_project/extracted_mot17/MOT17/test",
        output_dir="/root/autodl-tmp/HNN_A-COP/data/mot17_project/tianmouc_vid_proc/val",
        T_steps=40,
        is_test_set=True
    )