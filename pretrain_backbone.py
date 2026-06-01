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
    epoch_num = 100
    save_period = 1
    base_T_interval = 10  # 设定的两个真实相快门刷新帧之间的固定间隔 (T)
    
    save_ckpt_path = "ckpt/HNN_backbone.ckpt"
    Path(os.path.dirname(save_ckpt_path)).mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter('summary/pretrain/train_backbone_{}'.format(int(time.time())))
    
    backbone = TianmoucHNNBackbone().cuda()
    optimizer = torch.optim.Adam(backbone.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[50], gamma=0.1)
    criterion_feat = nn.MSELoss()
    
    image_dir = "tianmouc_data/pretrain_images"
    os.makedirs(image_dir, exist_ok=True)
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))
    
    if len(image_paths) == 0:
        print(f"⚠️ 未在 {image_dir} 找到大图，现自动注入 dummy 占位符进行管道测试。")
        image_paths = ["dummy_img_0.png", "dummy_img_1.png"] * 16
        
    train_dataset = TianmoucPretrainDataset(image_paths, base_T=base_T_interval)
    train_data = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    
    for epoch in tqdm(range(epoch_num)):
        backbone.train()
        for step, data in enumerate(train_data):
            cop_seq, td, sd = data
            
            # 虚拟数据防御隔离
            if "dummy_img" in image_paths[0]:
                cop_seq = torch.randn(1, 3, 320, 640, 42)
                td = torch.randn(1, 1, 160, 160, 42)
                sd = torch.randn(1, 2, 160, 160, 42)

            # 剥离 DataLoader 的 Batch 维，送入 GPU 高速演进
            cop_seq = cop_seq.cuda().squeeze(0)  # [3, 320, 640, T_random]
            td = td.cuda().squeeze(0)            # [1, 160, 160, T_random]
            sd = sd.cuda().squeeze(0)            # [2, 160, 160, T_random]
            
            # 一个长变向航迹序列开始时，重置一次 SNN 初始膜电位
            backbone.reset_stream_state()
            total_loss = 0
            T_random_total = td.shape[-1]  
            
            # 流式跨帧自回归训练
            for t in range(T_random_total):
                # 快门控制：只有在 base_T_interval 的倍数时间点，底图更新才生效
                is_rgb_available = (t % base_T_interval == 0)
                
                # 寻找当前区间对应的历史锚定快门帧索引
                current_snapshot_idx = (t // base_T_interval) * base_T_interval
                
                # 🚀 前向外推分支：网络输入端输入卡死在快门时刻的图像，在非快门时刻仅通过脉冲“盲操外推”隐状态
                current_feat = backbone(
                    rgb_frame=cop_seq[..., current_snapshot_idx].unsqueeze(0), 
                    sd_slice=sd[..., t].unsqueeze(0), 
                    td_slice=td[..., t].unsqueeze(0), 
                    is_rgb_available=is_rgb_available
                )
                
                # 🚀 【问题四完美修复】：获取当前微步移动后的真实图像，包裹在 no_grad 块内防止梯度乱飞与爆显存
                current_oracle_rgb = cop_seq[..., t].unsqueeze(0)
                with torch.no_grad():
                    oracle_feat_t = backbone.get_oracle_rgb_feature(current_oracle_rgb)
                
                # 损失计算：强迫非快门时刻的融合特征向当前步真实运动后的高阶语义看齐
                loss_step = criterion_feat(current_feat, oracle_feat_t)
                total_loss += loss_step
            
            total_loss = total_loss / T_random_total
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1)
            optimizer.step()

            writer.add_scalar('backbone pretrain alignment loss', total_loss.item(), step + 1 + epoch * len(train_data))

        scheduler.step()
        if (epoch + 1) % save_period == 0:
            print("\rsaving pretrained backbone at epoch {}, loss={:.3f}, path: {}".format(epoch + 1, total_loss.item(), save_ckpt_path))
            torch.save({
                'backbone': backbone.state_dict()
            }, save_ckpt_path)


if __name__ == "__main__":
    pretrain()