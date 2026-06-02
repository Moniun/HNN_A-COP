# /root/autodl-tmp/HNN_A-COP/city_to_tianmouc.py
import os
import json
import numpy as np
import cv2
import torch
from tqdm import tqdm
from tianmoucv.sim import run_sim_singleimg

def convert_city_to_tianmouc(img_dir, ann_dir, output_dir, T_steps=40, max_samples=15):
    os.makedirs(output_dir, exist_ok=True)
    
    # 搜集规整的图片列表
    img_files = []
    for root, dirs, files in os.walk(img_dir):
        for f in files:
            if f.endswith('_leftImg8bit.png'):
                img_files.append(os.path.join(root, f))
    img_files = sorted(img_files)
    
    print(f"🚀 Cityscapes 黄金战线全面点火！检测到有效标注图: {len(img_files)} 张。")
    
    for seq_idx, img_path in enumerate(tqdm(img_files[:max_samples])):
        # 寻找对应的 json 标签文件路径
        # 标准结构：把 leftImg8bit 换成 gtFine_polygons.json
        ann_path = img_path.replace('leftImg8bit', 'gtFine').replace('.png', '_polygons.json')
        
        # 1. 读取真实标签框
        raw_boxes = []
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                ann_data = json.load(f)
            for obj in ann_data.get('objects', []):
                # 过滤出你做 few-shot 最关心的 car 大类
                if obj.get('label') == 'car':
                    poly = np.array(obj['polygon'])
                    xmin, ymin = poly.min(axis=0)
                    xmax, ymax = poly.max(axis=0)
                    raw_boxes.append([xmin, ymin, xmax, ymax])
                    
        # 2. 执行几何运动仿真外推（生成 Pseudo 时序和真实连续 Labels）
        img_bgr = cv2.imread(img_path)
        H_orig, W_orig, _ = img_bgr.shape
        
        cop_frames_list = []
        labels_list = []
        
        # 模拟车辆轻微前移或视角颠簸（每帧产生固定平移扰动）
        dx_per_frame = 2.0  # 每帧向右漂移 2 像素
        dy_per_frame = 0.5  # 每帧向下颠簸 0.5 像素
        
        for t in range(T_steps):
            # 对原图执行仿射变换，模拟动态运动
            M = np.float32([[1, 0, t * dx_per_frame], [0, 1, t * dy_per_frame]])
            shifted_img = cv2.warpAffine(img_bgr, M, (W_orig, H_orig))
            img_rgb = cv2.cvtColor(shifted_img, cv2.COLOR_BGR2RGB)
            
            # 各向同性缩放至天眸标准的 320x640 视场
            img_resized = cv2.resize(img_rgb, (640, 320))
            cop_frames_list.append(img_resized)
            
            # 标签框跟随仿射变换同步平移，并做 320x640 坐标映射
            formatted_labels = np.zeros((3, 4), dtype=np.float32)
            for o_idx, box in enumerate(raw_boxes[:3]):
                xmin, ymin, xmax, ymax = box
                # 应用时序位移
                cx = ((xmin + xmax) / 2.0 + t * dx_per_frame) * (640.0 / W_orig)
                cy = ((ymin + ymax) / 2.0 + t * dy_per_frame) * (320.0 / H_orig)
                w = (xmax - xmin) * (640.0 / W_orig)
                h = (ymax - ymin) * (320.0 / H_orig)
                
                formatted_labels[o_idx] = [np.clip(cx, 0, 639), np.clip(cy, 0, 319), w, h]
            labels_list.append(formatted_labels)
            
        # 3. 喂入天眸双差分引擎榨取脉冲
        td_list, sd_list = [], []
        for t in range(T_steps):
            img_target = cop_frames_list[t]
            img_ref = cop_frames_list[t - 1] if t > 0 else None
            
            _, _, td_tensor, sd0_tensor, sd1_tensor = run_sim_singleimg(
                img_target=img_target, img_ref=img_ref,
                sensor_width=640, sensor_height=320, xy=False, interp=False, device=torch.device('cpu')
            )
            td_list.append(td_tensor.view(-1, 160, 160)[0].unsqueeze(0).numpy())
            sd_list.append(torch.cat([sd0_tensor.view(-1, 160, 160)[0].unsqueeze(0), sd1_tensor.view(-1, 160, 160)[0].unsqueeze(0)], dim=0).numpy())
            
        # 4. 固化持久化
        np.save(os.path.join(output_dir, f"seq_{seq_idx:03d}_cop_t0.npy"), cop_frames_list[0].transpose(2, 0, 1))
        np.save(os.path.join(output_dir, f"seq_{seq_idx:03d}_aop_td.npy"), np.stack(td_list, axis=-1))
        np.save(os.path.join(output_dir, f"seq_{seq_idx:03d}_aop_sd.npy"), np.stack(sd_list, axis=-1))
        np.save(os.path.join(output_dir, f"seq_{seq_idx:03d}_label.npy"), np.stack(labels_list, axis=-1))

    print(f"\n======================= 🎉 伪时序流式特征与真标签固化成功！ =======================")
    print(f"📂 真实有标签的特征矩阵已注入: {output_dir}")

if __name__ == "__main__":
    # 🔒 卡死更新后的绝对可读写路径
    convert_city_to_tianmouc(
        img_dir="/root/autodl-tmp/HNN_A-COP/data/cityscapes_project/extracted_leftImg8bit/leftImg8bit/train",
        ann_dir="/root/autodl-tmp/HNN_A-COP/data/cityscapes_project/extracted_gtFine/gtFine/train",
        output_dir="/root/autodl-tmp/HNN_A-COP/data/cityscapes_project/tianmouc_vid_proc/train",
        T_steps=40
    )