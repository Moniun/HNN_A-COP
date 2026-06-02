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
    epoch_num = 100
    save_period = 5  # 每 5 个 Epoch 固化保存一次权重
    
    # 🔒 路径管理
    load_backbone_path = "ckpt/HNN_backbone.ckpt"  # 载入你做完自监督训练的 Backbone 权重
    save_ckpt_path = "ckpt/HNN_detection_head.ckpt"
    Path(os.path.dirname(save_ckpt_path)).mkdir(parents=True, exist_ok=True)

    # 接入 TensorBoard 动态监控
    writer = SummaryWriter('summary/task_head/train_predictor_{}'.format(int(time.time())))
    
    # 1. 搭建拓扑架构毛坯房
    backbone = TianmoucHNNBackbone().cuda()
    task_head = TaskHead(in_channels=384, num_objects=3).cuda()
    
    # 2. 📡 强行注入预训练完备的大核自监督主干参数
    if load_backbone_path and os.path.exists(load_backbone_path):
        state_dict = torch.load(load_backbone_path, map_location=torch.device("cuda:0"))
        backbone.load_state_dict(state_dict.get('backbone', {}), strict=False)
        print(f"====== 📡 [Backbone 先验合体成功] 成功无缝加载预训练大核主干网络重量级参数 ======")
    else:
        print(f"⚠️ 提示：未在 '{load_backbone_path}' 下找到预训练权重，将采用常规初始化进行训练。")
    
    # 3. 🔒 科学隔离：冻结 Backbone 的所有参数，使其作为纯粹的流式差分特征提取器，绝不参与更新
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()  # 固化其内部 BN 层与注意力门控一阶惯性滤波的一阶低通统计量
    
    # 4. 科学优化：只把具有 requires_grad=True 的下游 TaskHead 核心参数送入 Adam 优化器
    # 并且将学习率下调至经典的学术稳健微调值 3e-4，防止梯度在 SNN 计算图上反复震荡爆炸
    trainable_params = [p for p in task_head.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=3e-4)
    
    # 加快学习率衰减进程，配合 100 个 Epoch 的短平快微调
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.2)
    criterion = nn.MSELoss()
    
    # 5. 极速无脑吃数：采用标准化离线落盘大底盘，CPU 零开销
    train_dataset = TianmoucStreamingDataset(test=False)
    train_data = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    
    print(f"🔥 全线点火！当前可调配 Few-Shot 轨迹大底盘片段数: {len(train_dataset)}")
    global_step = 0
    
    for epoch in range(epoch_num):
        task_head.train()  # 保持检测头处于激活状态
        pbar = tqdm(enumerate(train_data), total=len(train_data), desc=f"Epoch [{epoch+1}/{epoch_num}]")
        
        for step, data in pbar:
            cop, td, sd, cop_loc, target_loc = data
            cop, td, sd = cop.cuda(), td.cuda(), sd.cuda()
            target_loc = target_loc.cuda().float()
            
            # 每个 Batch 片段开局刷新神经元状态，迫使 Head 练习在零电位背景下瞬时咬死目标
            backbone.reset_stream_state()
            total_loss = 0
            T_steps = td.shape[-1]
            
            for t in range(T_steps):
                # 遵循因果流，片段内首帧加载 RGB 先验缓存，其余时步靠脉冲持续推演演进
                is_rgb_available = (t == 0)
                
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
            
            optimizer.zero_grad()
            mean_loss.backward()
            torch.nn.utils.clip_grad_norm_(task_head.parameters(), max_norm=1.0)
            optimizer.step()
            
            pbar.set_postfix({"MSE_Loss": f"{mean_loss.item():.4f}"})
            writer.add_scalar('training loss', mean_loss.item(), global_step)
            global_step += 1

        scheduler.step()
        
        # 固化成果
        if (epoch + 1) % save_period == 0 or (epoch + 1) == epoch_num:
            print(f"\n💾 [Epoch {epoch+1}] 成功固化微调成果, 当前 Loss={mean_loss.item():.4f}")
            torch.save({
                'backbone': backbone.state_dict(),
                'head': task_head.state_dict()
            }, save_ckpt_path)

if __name__ == "__main__":
    train()