from dataset import TianmoucPretrainDataset 
from models import TianmoucHNNBackbone
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import time
import os
import glob


def inject_extreme_noise_to_batch(batch_img_tensor, prob=0.3):
    """
    🚀 批次级初始帧环境破坏器：
    仅在时序最初始触发点，随机模拟出隧道突发强光过载过曝，或者高强度传感器高频噪声
    输入 batch_img_tensor 形状: [B, 3, H, W] (值域 0~1)
    """
    if np.random.rand() > prob:
        return batch_img_tensor # 保持正常干净环境
        
    b, c, h, w = batch_img_tensor.shape
    noisy_batch = batch_img_tensor.clone()
    
    aug_type = np.random.choice(['flash_overexposure', 'sensor_gaussian_noise'])
    
    if aug_type == 'flash_overexposure':
        # 模拟出隧道口或前向突发强眩目闪光：全局亮度大范围飙升饱和
        flash_bias = np.random.uniform(0.4, 0.7)
        noisy_batch = noisy_batch + flash_bias
        noisy_batch = torch.clamp(noisy_batch, 0.0, 1.0)
        
    elif aug_type == 'sensor_gaussian_noise':
        # 模拟极端低照度环境或芯片过热引发的高频高斯白噪声
        sigma = np.random.uniform(0.05, 0.15)
        noise = torch.randn_like(noisy_batch) * sigma
        noisy_batch = noisy_batch + noise
        noisy_batch = torch.clamp(noisy_batch, 0.0, 1.0)
        
    return noisy_batch


def pretrain():
    epoch_num = 5
    save_period = 1
    base_T_interval = 10  
    
    save_ckpt_path = "ckpt/HNN_backbone.ckpt"
    Path(os.path.dirname(save_ckpt_path)).mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter('summary/pretrain/train_backbone_{}'.format(int(time.time())))
    
    backbone = TianmoucHNNBackbone().cuda()
    trainable_params = [p for p in backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[50], gamma=0.1)
    criterion_feat = nn.MSELoss()
    
    image_dir = "tianmouc_data/pretrain_images"
    os.makedirs(image_dir, exist_ok=True)
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))
    
    if len(image_paths) == 0:
        raise FileNotFoundError(f"❌ 错误：未在目录 '{image_dir}' 下找到任何 .jpg 或 .png 图像，请放入真实大图后再运行！")
        
    train_dataset = TianmoucPretrainDataset(image_paths, base_T=base_T_interval)
    train_data = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    
    global_step = 0
    for epoch in range(epoch_num):
        backbone.train()
        pbar = tqdm(enumerate(train_data), total=len(train_data), desc=f"Epoch [{epoch+1}/{epoch_num}]")
        
        for step, data in pbar:
            cop_seq, td, sd = data
            cop_seq = cop_seq.cuda()  
            td = td.cuda()            
            sd = sd.cuda()            
            
            backbone.reset_stream_state()
            total_loss = 0
            total_cos_sim = 0 
            T_random_total = td.shape[-1]  
            
            for t in range(T_random_total):
                is_rgb_available = (t % base_T_interval == 0)
                current_snapshot_idx = (t // base_T_interval) * base_T_interval
                
                # 提取当前的基准输入帧 [B, 3, H, W]
                input_rgb_frame = cop_seq[..., current_snapshot_idx]
                
                # 🚀【这才是完全正确的噪声注入位置】：
                # 只有当处于快门刷新起点 (t % base_T_interval == 0) 且真正喂入可训练主路时，
                # 我们才对其注入出隧道过载等Corner Case噪声。
                # 其余盲操时间段，网络不接收任何 COP 图像输入。
                if is_rgb_available:
                    input_rgb_frame = inject_extreme_noise_to_batch(input_rgb_frame, prob=0.3)
                
                current_feat = backbone(
                    rgb_frame=input_rgb_frame, 
                    sd_slice=sd[..., t], 
                    td_slice=td[..., t], 
                    is_rgb_available=is_rgb_available
                )
                
                # 🔒【教师侧绝对保护】：
                # 教师路由 get_oracle_rgb_feature 永远吃进 100% 干净保真的当前移动帧，
                # 提供毫无偏置污染、最纯净、最高阶的空间语义蒸馏标杆！
                current_oracle_rgb = cop_seq[..., t]
                with torch.no_grad():
                    oracle_feat_t = backbone.get_oracle_rgb_feature(current_oracle_rgb)
                
                loss_step = criterion_feat(current_feat, oracle_feat_t)
                total_loss += loss_step
                
                with torch.no_grad():
                    step_cos = torch.nn.functional.cosine_similarity(current_feat, oracle_feat_t, dim=1).mean()
                    total_cos_sim += step_cos
            
            mean_loss = total_loss / T_random_total
            mean_cos = total_cos_sim / T_random_total
            
            optimizer.zero_grad()
            mean_loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1)
            optimizer.step()

            pbar.set_postfix({
                "MSE_Loss": f"{mean_loss.item():.4f}",
                "Cos_Sim": f"{mean_cos.item():.4f}" 
            })

            if global_step % 100 == 0:
                with torch.no_grad():
                    pred_heatmap = torch.mean(current_feat[0:1], dim=1, keepdim=True)
                    oracle_heatmap = torch.mean(oracle_feat_t[0:1], dim=1, keepdim=True)
                    
                    pred_heatmap = (pred_heatmap - pred_heatmap.min()) / (pred_heatmap.max() - pred_heatmap.min() + 1e-5)
                    oracle_heatmap = (oracle_heatmap - oracle_heatmap.min()) / (oracle_heatmap.max() - oracle_heatmap.min() + 1e-5)
                    
                    writer.add_image('Visual/1_Predicted_HNN_Out推演', pred_heatmap[0], global_step)
                    writer.add_image('Visual/2_Oracle_GT_大模型真值', oracle_heatmap[0], global_step)

            writer.add_scalar('pretrain/MSE_Loss', mean_loss.item(), global_step)
            writer.add_scalar('pretrain/Cosine_Similarity', mean_cos.item(), global_step)
            global_step += 1

        scheduler.step()
        if (epoch + 1) % save_period == 0:
            print(f"\n[Epoch {epoch+1}] saving pretrained backbone, loss={mean_loss.item():.3f}")
            torch.save({'backbone': backbone.state_dict()}, save_ckpt_path)


if __name__ == "__main__":
    pretrain()