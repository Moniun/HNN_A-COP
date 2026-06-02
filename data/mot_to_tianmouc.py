# /root/autodl-tmp/HNN_A-COP/data/mot_to_tianmouc.py
import os
import glob
import numpy as np
import cv2
import torch
from tqdm import tqdm
from tianmoucv.sim import run_sim_singleimg

def parse_gt_txt(gt_path):
    frame_dict = {}
    if not os.path.exists(gt_path): return frame_dict
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7: continue
            frame_id, class_id = int(parts[0]), int(parts[7])
            if class_id not in [1, 3]: continue  # 只提纯行人和车辆
            x1, y1, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            frame_dict.setdefault(frame_id, []).append([x1 + w/2.0, y1 + h/2.0, w, h])
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
        full_video_cop, full_video_label = [], []
        
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
                    # 🔒 核心修复 1：在离线端直接完成 [0, 1] 全量各项同性相对坐标归一化
                    formatted_labels[o_idx] = [
                        np.clip(cx / W_orig, 0, 1),
                        np.clip(cy / H_orig, 0, 1),
                        w / W_orig,
                        h / H_orig
                    ]
            full_video_label.append(formatted_labels)
            
        print(f"⚡ 正在应用官方视频流因果逻辑，榨取高保真 DVS 脉冲特征...")
        full_td_list, full_sd_list = [], []
        for t in range(len(all_imgs)):
            img_target = full_video_cop[t]
            img_ref = full_video_cop[t - 1] if t > 0 else None 
            
            _, _, td_tensor, sd0_tensor, sd1_tensor = run_sim_singleimg(
                img_target=img_target, img_ref=img_ref,
                sensor_width=640, sensor_height=320, xy=False, interp=True, device=torch.device('cpu')
            )
            full_td_list.append(td_tensor.view(-1, 160, 160)[0].unsqueeze(0).numpy())
            full_sd_list.append(torch.cat([sd0_tensor.view(-1, 160, 160)[0].unsqueeze(0), sd1_tensor.view(-1, 160, 160)[0].unsqueeze(0)], dim=0).numpy())
            
        full_cop_arr = np.stack(full_video_cop, axis=0).transpose(3, 1, 2, 0)
        full_td_arr = np.stack(full_td_list, axis=-1)
        full_sd_arr = np.stack(full_sd_list, axis=-1)
        full_label_arr = np.stack(full_video_label, axis=-1)
        
        # 3. 后端存储切块
        num_segments = len(all_imgs) // T_steps
        for seg_idx in range(num_segments):
            s_t, e_t = seg_idx * T_steps, seg_idx * T_steps + T_steps
            
            segment_labels = full_label_arr[..., s_t:e_t]
            
            # 🔒 核心修复 2：如果不是测试集，并且当前连续 40 帧内没有任何有效的目标框（绝对和全零一致）
            # 直接抛弃，绝不写入硬盘污染网络！
            if not is_test_set and np.allclose(segment_labels, 0):
                continue
                
            np.save(os.path.join(output_dir, f"{seq_name}_seg_{seg_idx:02d}_cop_t0.npy"), full_cop_arr[..., s_t:e_t])
            np.save(os.path.join(output_dir, f"{seq_name}_seg_{seg_idx:02d}_aop_td.npy"), full_td_arr[..., s_t:e_t])
            np.save(os.path.join(output_dir, f"{seq_name}_seg_{seg_idx:02d}_aop_sd.npy"), full_sd_arr[..., s_t:e_t])
            np.save(os.path.join(output_dir, f"{seq_name}_seg_{seg_idx:02d}_label.npy"), segment_labels)
            global_seq_idx += 1

    print(f"\n======================= 🎉 Standard MOT17 绝对连续大底盘落盘成功！ =======================")

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