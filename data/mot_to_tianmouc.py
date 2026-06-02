import os
import glob
import numpy as np
import cv2
import torch
from tqdm import tqdm
from tianmoucv.sim import run_sim_singleimg

def parse_gt_txt(gt_path):
    """高效解析 MOT17 原生真实时序标签"""
    frame_dict = {}
    if not os.path.exists(gt_path):
        return frame_dict
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7: continue
            frame_id = int(parts[0])
            class_id = int(parts[7])
            if class_id not in [1, 3]: continue  # 只提纯行人和车辆
            
            x1, y1, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0
            frame_dict.setdefault(frame_id, []).append([cx, cy, w, h])
    return frame_dict

def convert_mot_to_tianmouc(src_base_dir, output_dir, T_steps=40, is_test_set=False):
    os.makedirs(output_dir, exist_ok=True)
    
    seq_paths = sorted([d for d in glob.glob(os.path.join(src_base_dir, "*")) if os.path.isdir(d)])
    mode_str = "【测试集-无标签推理模式】" if is_test_set else "【训练集-带真值微调模式】"
    print(f"🚀 开启 MOT17 {mode_str} 物理视频流无缝仿真固化系统...")
    
    for seq_path in seq_paths:
        seq_name = os.path.basename(seq_path)
        img_dir = os.path.join(seq_path, "img1")
        gt_path = os.path.join(seq_path, "gt", "gt.txt")
        
        all_imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        gt_dict = parse_gt_txt(gt_path)
        
        print(f"🎬 正在全量加载视频序列并执行 640x320 各向同性映射: {seq_name} (共 {len(all_imgs)} 帧)")
        full_video_cop = []
        full_video_label = []
        
        # 1. 预先一次性加载整段视频，完成分辨率缩放与框坐标缩放映射
        for frame_idx, img_path in enumerate(all_imgs):
            frame_bgr = cv2.imread(img_path)
            H_orig, W_orig, _ = frame_bgr.shape
            frame_resized = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), (640, 320))
            full_video_cop.append(frame_resized)
            
            formatted_labels = np.zeros((3, 4), dtype=np.float32)
            if not is_test_set:
                raw_boxes = gt_dict.get(frame_idx + 1, [])
                for o_idx, box in enumerate(raw_boxes[:3]):
                    cx, cy, w, h = box
                    formatted_labels[o_idx] = [
                        np.clip(cx * (640.0 / W_orig), 0, 639),
                        np.clip(cy * (320.0 / H_orig), 0, 319),
                        w * (640.0 / W_orig),
                        h * (320.0 / H_orig)
                    ]
            full_video_label.append(formatted_labels)
            
        # 2. 🔒 黄金对齐：对整个完整长视频无脑一键跑天眸仿真，榨取绝对物理连续的脉冲大矩阵！
        print(f"⚡ 正在应用官方视频流因果逻辑，榨取高保真 DVS 脉冲特征...")
        full_td_list, full_sd_list = [], []
        for t in range(len(all_imgs)):
            img_target = full_video_cop[t]
            # 第 0 帧没有参考帧（标准暗电流开机），从第 1 帧起死死参考 t-1 帧
            img_ref = full_video_cop[t - 1] if t > 0 else None 
            
            _, _, td_tensor, sd0_tensor, sd1_tensor = run_sim_singleimg(
                img_target=img_target,
                img_ref=img_ref,
                sensor_width=640,
                sensor_height=320,
                xy=False,
                interp=True,  # 开启上采样，对齐 640x320
                device=torch.device('cpu')
            )
            full_td_list.append(td_tensor.view(-1, 160, 160)[0].unsqueeze(0).numpy())
            full_sd_list.append(torch.cat([sd0_tensor.view(-1, 160, 160)[0].unsqueeze(0), sd1_tensor.view(-1, 160, 160)[0].unsqueeze(0)], dim=0).numpy())
            
        # 组装时序完整的大序列
        full_cop_arr = np.stack(full_video_cop, axis=0).transpose(3, 1, 2, 0) # [3, 320, 640, Total_Frames]
        full_td_arr = np.stack(full_td_list, axis=-1)                        # [1, 160, 160, Total_Frames]
        full_sd_arr = np.stack(full_sd_list, axis=-1)                        # [2, 160, 160, Total_Frames]
        full_label_arr = np.stack(full_video_label, axis=-1)                  # [3, 4, Total_Frames]
        
        # 3. 🔒 在后端存储层，切分成标准的 T_steps=40 独立小片段持久化落盘
        num_segments = len(all_imgs) // T_steps
        for seg_idx in range(num_segments):
            s_t = seg_idx * T_steps
            e_t = s_t + T_steps
            
            # 采用 seq_name 建立物理命名隔离，方便 test() 识别是否更换视频
            np.save(os.path.join(output_dir, f"{seq_name}_seg_{seg_idx:02d}_cop_t0.npy"), full_cop_arr[..., s_t:e_t])
            np.save(os.path.join(output_dir, f"{seq_name}_seg_{seg_idx:02d}_aop_td.npy"), full_td_arr[..., s_t:e_t])
            np.save(os.path.join(output_dir, f"{seq_name}_seg_{seg_idx:02d}_aop_sd.npy"), full_sd_arr[..., s_t:e_t])
            np.save(os.path.join(output_dir, f"{seq_name}_seg_{seg_idx:02d}_label.npy"), full_label_arr[..., s_t:e_t])

    print(f"\n======================= 🎉 MOT17 全视频串联预处理圆满成功！ =======================")

if __name__ == "__main__":
    # 🛠️ 第一阶段点火：全量洗出训练集（有标签，进 train 目录）
    convert_mot_to_tianmouc(
        src_base_dir="/root/autodl-tmp/HNN_A-COP/data/mot17_project/extracted_mot17/MOT17/train",
        output_dir="/root/autodl-tmp/HNN_A-COP/data/mot17_project/tianmouc_vid_proc/train",
        T_steps=25,
        is_test_set=False
    )
    
    # 🛠️ 第二阶段点火：全量洗出测试集（无标签，进 val 目录，无缝匹配 DataLoader）
    convert_mot_to_tianmouc(
        src_base_dir="/root/autodl-tmp/HNN_A-COP/data/mot17_project/extracted_mot17/MOT17/test",
        output_dir="/root/autodl-tmp/HNN_A-COP/data/mot17_project/tianmouc_vid_proc/val",
        T_steps=25,
        is_test_set=True
    )