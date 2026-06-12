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
import torch.nn.functional as F


def inject_extreme_noise_to_batch(batch_img_tensor, prob=0.3):
    if np.random.rand() > prob:
        return batch_img_tensor 
    b, c, h, w = batch_img_tensor.shape
    noisy_batch = batch_img_tensor.clone()
    aug_type = np.random.choice(['flash_overexposure', 'sensor_gaussian_noise'])
    if aug_type == 'flash_overexposure':
        flash_bias = np.random.uniform(0.4, 0.7)
        noisy_batch = noisy_batch + flash_bias
        noisy_batch = torch.clamp(noisy_batch, 0.0, 1.0)
    elif aug_type == 'sensor_gaussian_noise':
        sigma = np.random.uniform(0.05, 0.15)
        noise = torch.randn_like(noisy_batch) * sigma
        noisy_batch = noisy_batch + noise
        noisy_batch = torch.clamp(noisy_batch, 0.0, 1.0)
    return noisy_batch


def pretrain():
    epoch_num = 30
    save_period = 1
    base_T_interval = 10  
    
    save_ckpt_path = "ckpt/HNN_backbone.ckpt"
    best_ckpt_path = "ckpt/HNN_backbone_best.ckpt"
    Path(os.path.dirname(save_ckpt_path)).mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter('summary/pretrain/train_backbone_{}'.format(int(time.time())))
    
    backbone = TianmoucHNNBackbone().cuda()
    trainable_params = [p for p in backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=5e-4)  # 降低学习率
    scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.2)  # 调整学习率衰减时机
    criterion_feat = nn.MSELoss()
    
    image_dir = "data/pretrain_images"
    os.makedirs(image_dir, exist_ok=True)
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))
    
    if len(image_paths) == 0:
        raise FileNotFoundError(f"❌ 错误：未在目录 '{image_dir}' 下找到任何 .jpg 或 .png 图像，请放入真实大图后再运行！")
        
    train_dataset = TianmoucPretrainDataset(image_paths, base_T=base_T_interval)
    train_data = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    
    global_step = 0
    # 🏆 跟踪最佳性能指标（使用 Cos Similarity，越大越好）
    best_cos_sim = -float('inf')
    best_epoch = 0
    
    print(f"\n{'='*80}")
    print(f"{'🚀 Backbone 预训练启动':^80}")
    print(f"{'='*80}")
    print(f"📊 配置信息:")
    print(f"   - Epoch 总数: {epoch_num}")
    print(f"   - 数据集大小: {len(train_dataset)} 样本")
    print(f"   - Batch Size: {train_data.batch_size}")
    print(f"   - 学习率: {optimizer.param_groups[0]['lr']}")
    print(f"   - 保存周期: 每 {save_period} 个 Epoch")
    print(f"   - 最优模型保存到: {best_ckpt_path}")
    print(f"{'='*80}\n")
    
    for epoch in range(epoch_num):
        backbone.train()
        
        # 📊 Epoch 级别的统计信息
        epoch_total_loss = 0.0
        epoch_total_cos = 0.0
        epoch_step_count = 0
        
        # 创建进度条，显示 Epoch 进度
        pbar = tqdm(enumerate(train_data), total=len(train_data), 
                    desc=f"📈 Epoch {epoch+1:2d}/{epoch_num:2d}")
        
        for step, data in pbar:
            cop_seq, td, sd = data
            cop_seq = cop_seq.cuda()  
            td = td.cuda()            
            sd = sd.cuda()            
            
            backbone.reset_stream_state()
            total_loss = 0
            total_cos_sim = 0 
            T_random_total = td.shape[-1]  
            
            # 🔒 核心优化：创建时域连续性缓存指针
            last_step_feat = None
            
            for t in range(T_random_total):
                is_rgb_available = (t % base_T_interval == 0)
                current_snapshot_idx = (t // base_T_interval) * base_T_interval
                
                input_rgb_frame = cop_seq[..., current_snapshot_idx]
                if is_rgb_available:
                    input_rgb_frame = inject_extreme_noise_to_batch(input_rgb_frame, prob=0.3)
                
                current_feat = backbone(
                    rgb_frame=input_rgb_frame, 
                    sd_slice=sd[..., t], 
                    td_slice=td[..., t], 
                    is_rgb_available=is_rgb_available
                )
                
                current_oracle_rgb = cop_seq[..., t]
                with torch.no_grad():
                    oracle_feat_t = backbone.get_oracle_rgb_feature(current_oracle_rgb)
                
                # 特征归一化（标准化到单位范数）
                current_feat_norm = F.normalize(current_feat, dim=1)
                oracle_feat_norm = F.normalize(oracle_feat_t, dim=1)
                
                # 1. 基础特征对齐损失（使用归一化特征）
                loss_step = criterion_feat(current_feat_norm, oracle_feat_norm)
                
                # 2. 余弦相似度损失（辅助对齐方向）
                cos_sim = F.cosine_similarity(current_feat_norm, oracle_feat_norm, dim=1).mean()
                cos_loss = 1 - cos_sim  # 转换为损失，越小越好
                
                # 3. 时域连续性正则项（惩罚前后特征的病态剧烈跳变）
                temporal_smooth_loss = 0
                if last_step_feat is not None:
                    last_feat_norm = F.normalize(last_step_feat, dim=1)
                    temporal_smooth_loss = criterion_feat(current_feat_norm, last_feat_norm)
                
                # 综合损失：主损失 + 余弦损失 + 时域正则
                loss_step = 2.0 * loss_step + cos_loss + 0.5 * temporal_smooth_loss
                
                last_step_feat = current_feat.clone().detach()
                total_loss += loss_step
                
                with torch.no_grad():
                    step_cos = torch.nn.functional.cosine_similarity(current_feat, oracle_feat_t, dim=1).mean()
                    total_cos_sim += step_cos
            
            mean_loss = total_loss / T_random_total
            mean_cos = total_cos_sim / T_random_total
            
            # 📊 累加 Epoch 统计
            epoch_total_loss += mean_loss.item()
            epoch_total_cos += mean_cos.item()
            epoch_step_count += 1
            
            optimizer.zero_grad()
            mean_loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1)
            optimizer.step()

            # 更新进度条显示（实时值 + 当前 Epoch 平均值）
            epoch_avg_loss = epoch_total_loss / epoch_step_count
            epoch_avg_cos = epoch_total_cos / epoch_step_count
            pbar.set_postfix({
                "Loss": f"{mean_loss.item():.8f}",
                "Avg_Loss": f"{epoch_avg_loss:.8f}",
                "CosSim": f"{mean_cos.item():.8f}",
                "Avg_CosSim": f"{epoch_avg_cos:.8f}"
            })

            if global_step % 100 == 0:
                with torch.no_grad():
                    pred_heatmap = torch.mean(current_feat[0:1], dim=1, keepdim=True)
                    oracle_heatmap = torch.mean(oracle_feat_t[0:1], dim=1, keepdim=True)
                    
                    pred_heatmap = (pred_heatmap - pred_heatmap.min()) / (pred_heatmap.max() - pred_heatmap.min() + 1e-5)
                    oracle_heatmap = (oracle_heatmap - oracle_heatmap.min()) / (oracle_heatmap.max() - oracle_heatmap.min() + 1e-5)
                    
                    writer.add_image('Visual/1_Predicted_HNN_Out推推演', pred_heatmap[0], global_step)
                    writer.add_image('Visual/2_Oracle_GT_大模型真值', oracle_heatmap[0], global_step)

            writer.add_scalar('pretrain/MSE_Loss', mean_loss.item(), global_step)
            writer.add_scalar('pretrain/Cosine_Similarity', mean_cos.item(), global_step)
            global_step += 1

        scheduler.step()
        
        # 📊 Epoch 结束时打印详细统计
        epoch_avg_loss = epoch_total_loss / epoch_step_count
        epoch_avg_cos = epoch_total_cos / epoch_step_count
        
        print(f"\n{'='*80}")
        print(f"✅ Epoch {epoch+1:2d}/{epoch_num:2d} 完成！")
        print(f"📊 Epoch 平均指标:")
        print(f"   - 平均 MSE Loss: {epoch_avg_loss:.6f}")
        print(f"   - 平均 Cos Similarity: {epoch_avg_cos:.6f}")
        
        # 🏆 检查是否是最佳模型
        is_best = epoch_avg_cos > best_cos_sim
        if is_best:
            best_cos_sim = epoch_avg_cos
            best_epoch = epoch + 1
            print(f"🏆 发现最佳模型！Cos Similarity: {best_cos_sim:.6f} (Epoch {best_epoch})")
            torch.save({'backbone': backbone.state_dict()}, best_ckpt_path)
        
        # 定期保存模型（可选，保留最后一个 Epoch）
        if (epoch + 1) % save_period == 0:
            print(f"💾 保存最新模型到: {save_ckpt_path}")
            torch.save({'backbone': backbone.state_dict()}, save_ckpt_path)
        
        print(f"\n📊 当前最佳记录: Epoch {best_epoch} | Cos Similarity: {best_cos_sim:.6f}")
        print(f"{'='*80}\n")
    
    print(f"\n{'='*80}")
    print(f"🎉 训练完成！")
    print(f"🏆 最佳模型: Epoch {best_epoch} | Cos Similarity: {best_cos_sim:.6f}")
    print(f"💾 最优模型已保存到: {best_ckpt_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    pretrain()