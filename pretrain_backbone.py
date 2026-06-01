from pretrain_dataset import TianmoucPretrainDataset
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
import glob


def pretrain():
    epoch_num = 100
    save_period = 1
    base_T_interval = 10  # 💡 核心：你设想的两个真实快门帧之间的固定间隔 (T)
    
    save_ckpt_path = "ckpt/HNN_backbone.ckpt"
    Path(os.path.dirname(save_ckpt_path)).mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter('summary/pretrain/train_backbone_{}'.format(int(time.time())))
    
    backbone = TianmoucHNNBackbone().cuda()
    optimizer = torch.optim.Adam(backbone.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[5], gamma=0.1)
    criterion_feat = nn.MSELoss()
    
    image_dir = "tianmouc_data/pretrain_images"
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))
    if len(image_paths) == 0:
        image_paths = ["dummy_img_0.png", "dummy_img_1.png"] * 16
        
    train_dataset = TianmoucPretrainDataset(image_paths, base_T=base_T_interval)
    
    # 💡 核心改动：由于每个 Item 生成的总时步长度是完全随机的，
    # 我们设置 batch_size=1，让网络单流高频演进，完美避免了变长张量无法打包的问题
    train_data = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    
    for epoch in tqdm(range(epoch_num)):
        backbone.train()
        for step, data in enumerate(train_data):
            # cop_seq 形状: [1, 3, 320, 640, T_random]
            # td 形状: [1, 1, 160, 160, T_random]
            cop_seq, td, sd = data
            cop_seq = cop_seq.cuda().squeeze(0)  # 剥离 Batch 维 -> [3, 320, 640, T_random]
            td = td.cuda().squeeze(0)            # 剥离 Batch 维 -> [1, 160, 160, T_random]
            sd = sd.cuda().squeeze(0)            # 剥离 Batch 维 -> [2, 160, 160, T_random]
            
            # 在一个随机超长序列开始时，清空一次全局膜电位和旧底图记忆
            backbone.reset_stream_state()
            total_loss = 0
            T_random_total = td.shape[-1]  # 获取本次 Item 随机生成的总步长 (40~80之间)
            
            # 全时步流式自回归演进开始
            for t in range(T_random_total):
                # 🚀 跨快门帧灵活锚定逻辑：
                # 只有在 t 能被 base_T_interval 整除时（如 t=0, 10, 20, 30...），新的真实 COP 图像才到达
                is_rgb_available = (t % base_T_interval == 0)
                
                # 计算当前时间步应该寻找哪一个历史快门帧作为前向基础输入：
                # 比如 t=0~9 步时，网络始终输入第 0 帧；t=10~19 步时，输入最新的第 10 帧，以此类推！
                current_snapshot_idx = (t // base_T_interval) * base_T_interval
                
                # 💡 前向盲操推进：即使中间底图发生了刷新，SNN 的状态也在 backbone 内部持续沿用，绝不重置
                current_feat = backbone(
                    rgb_frame=cop_seq[..., current_snapshot_idx].unsqueeze(0), # 补充 Batch 维喂入网络 
                    sd_slice=sd[..., t].unsqueeze(0), 
                    td_slice=td[..., t].unsqueeze(0), 
                    is_rgb_available=is_rgb_available
                )
                
                # 🚀 获取当前时步的教师特征目标（包含真实的复杂多阶段平移位置）
                current_oracle_rgb = cop_seq[..., t].unsqueeze(0)
                with torch.no_grad():
                    oracle_feat_t = backbone.cop_net.step(current_oracle_rgb)
                
                # 强监督对齐：迫使融合特征去预测并修正当前时刻真实的运动形态
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