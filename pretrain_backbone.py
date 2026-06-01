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


def pretrain():
    epoch_num = 10
    save_period = 1
    base_T_interval = 10  # 两个真实相快门刷新帧之间的固定间隔 (T)
    
    save_ckpt_path = "ckpt/HNN_backbone.ckpt"
    Path(os.path.dirname(save_ckpt_path)).mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter('summary/pretrain/train_backbone_{}'.format(int(time.time())))
    
    backbone = TianmoucHNNBackbone().cuda()
    
    # 🚀 修正 1：过滤掉被冻结的大模型参数，仅将可训练的 SNN 路径和 HUs 单元送入优化器
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
    train_data = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    
    for epoch in range(epoch_num):
        backbone.train()
        
        # 移至内层的全量高频滚动进度条
        pbar = tqdm(enumerate(train_data), total=len(train_data), desc=f"Epoch [{epoch+1}/{epoch_num}]")
        
        for step, data in pbar:
            cop_seq, td, sd = data

            cop_seq = cop_seq.cuda().squeeze(0)  
            td = td.cuda().squeeze(0)            
            sd = sd.cuda().squeeze(0)            
            
            backbone.reset_stream_state()
            total_loss = 0
            T_random_total = td.shape[-1]  
            
            for t in range(T_random_total):
                is_rgb_available = (t % base_T_interval == 0)
                current_snapshot_idx = (t // base_T_interval) * base_T_interval
                
                current_feat = backbone(
                    rgb_frame=cop_seq[..., current_snapshot_idx].unsqueeze(0), 
                    sd_slice=sd[..., t].unsqueeze(0), 
                    td_slice=td[..., t].unsqueeze(0), 
                    is_rgb_available=is_rgb_available
                )
                
                current_oracle_rgb = cop_seq[..., t].unsqueeze(0)
                with torch.no_grad():
                    oracle_feat_t = backbone.get_oracle_rgb_feature(current_oracle_rgb)
                
                # 🚀 修正 2：去除了死板的硬对齐强拉伸，在纯净的 [384, 20, 40] 隐空间发生特征蒸馏
                loss_step = criterion_feat(current_feat, oracle_feat_t)
                total_loss += loss_step
            
            total_loss = total_loss / T_random_total
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1)
            optimizer.step()

            pbar.set_postfix({"Step_Loss": f"{total_loss.item():.4f}"})
            writer.add_scalar('backbone pretrain alignment loss', total_loss.item(), step + 1 + epoch * len(train_data))

        scheduler.step()
        if (epoch + 1) % save_period == 0:
            print(f"\n[Epoch {epoch+1}] saving pretrained backbone, loss={total_loss.item():.3f}")
            torch.save({'backbone': backbone.state_dict()}, save_ckpt_path)


if __name__ == "__main__":
    pretrain()