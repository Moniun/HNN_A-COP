# /root/autodl-tmp/HNN_A-COP/train.py
import os
import time
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from dataset.tmdat_dataset import TianmoucStreamingDataset
from models import TianmoucHNNBackbone, TaskHead

def train():
    epoch_num = 50
    save_period = 5  
    base_T_interval = 10  # 🔒 核心对齐：锁死 10 帧为周期的流式门控注意力唤醒频率！
    
    load_backbone_path = "ckpt/HNN_backbone.ckpt"  
    save_ckpt_path = "ckpt/HNN_detection_head.ckpt"
    Path(os.path.dirname(save_ckpt_path)).mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter('summary/task_head/train_predictor_{}'.format(int(time.time())))
    
    backbone = TianmoucHNNBackbone().cuda()
    task_head = TaskHead(in_channels=384, num_objects=3).cuda()
    
    if load_backbone_path and os.path.exists(load_backbone_path):
        state_dict = torch.load(load_backbone_path, map_location=torch.device("cuda:0"))
        backbone.load_state_dict(state_dict.get('backbone', {}), strict=False)
        print(f"====== 📡 [Backbone 先验合体成功] 成功无缝加载预训练大核主干网络重量级参数 ======")
    
    # 冻结backbone网络
    # for param in backbone.parameters():
    #     param.requires_grad = False
    # backbone.eval()  
    
    trainable_params = [p for p in task_head.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=3e-4) # 恢复到标准的 3e-4 微调黄金学习率
    
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.2)
    criterion = nn.MSELoss()
    
    train_dataset = TianmoucStreamingDataset(test=False)
    train_data = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    
    print(f"🔥 全线点火！当前可调配 Few-Shot 真实轨迹片段数: {len(train_dataset)}")
    global_step = 0
    
    for epoch in range(epoch_num):
        task_head.train()  
        pbar = tqdm(enumerate(train_data), total=len(train_data), desc=f"Epoch [{epoch+1}/{epoch_num}]")
        
        epoch_total_loss = 0.0  # 记录整个epoch的总loss
        epoch_step_count = 0     # 记录epoch内的step数
        
        for step, data in pbar:
            cop, td, sd, cop_loc, target_loc = data
            cop, td, sd = cop.cuda(), td.cuda(), sd.cuda()
            target_loc = target_loc.cuda().float()
            
            backbone.reset_stream_state()
            total_loss = 0
            T_steps = td.shape[-1]
            
            for t in range(T_steps):
                # 🔒 黄金对齐防线：每隔 10 帧让 is_rgb_available 弹起为 True，更新当前的动态街景先验！
                # 其余时步（t % 10 != 0 时）网络依靠完美的、无空档的 AOP 通路脉冲实现高动态连续位移自回归！
                is_rgb_available = (t % base_T_interval == 0)
                
                current_feat = backbone(
                    rgb_frame=cop[..., t], 
                    sd_slice=sd[..., t], 
                    td_slice=td[..., t], 
                    is_rgb_available=is_rgb_available
                )
                
                pred_loc = task_head(current_feat)
                loss_step = criterion(pred_loc, target_loc[..., t])
                total_loss += loss_step
            
            mean_loss = total_loss / T_steps
            
            # 累积到epoch总loss
            epoch_total_loss += mean_loss.item()
            epoch_step_count += 1
            
            optimizer.zero_grad()
            mean_loss.backward()
            torch.nn.utils.clip_grad_norm_(task_head.parameters(), max_norm=1.0)
            optimizer.step()
            
            pbar.set_postfix({"MSE_Loss": f"{mean_loss.item():.4f}"})
            writer.add_scalar('training loss', mean_loss.item(), global_step)
            global_step += 1

        scheduler.step()
        
        # 📊 计算并显示当前epoch的平均loss
        epoch_avg_loss = epoch_total_loss / epoch_step_count
        print(f"\n📊 [Epoch {epoch+1}/{epoch_num}] 训练完成 | 平均 MSE Loss = {epoch_avg_loss:.4f}")
        
        if (epoch + 1) % save_period == 0 or (epoch + 1) == epoch_num:
            print(f"💾 [Epoch {epoch+1}] 成功保存微调成果, 当前 Loss={mean_loss.item():.4f}")
            torch.save({
                'backbone': backbone.state_dict(),
                'head': task_head.state_dict()
            }, save_ckpt_path)

if __name__ == "__main__":
    train()