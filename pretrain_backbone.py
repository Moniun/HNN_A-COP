from dataset.pretrain_dataset import TianmoucPretrainDataset
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
    save_ckpt_path = "ckpt/HNN_backbone.ckpt"
    Path(os.path.dirname(save_ckpt_path)).mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter('summary/pretrain/train_backbone_{}'.format(int(time.time())))
    
    backbone = TianmoucHNNBackbone().cuda()
    
    optimizer = torch.optim.Adam(backbone.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[5], gamma=0.1)
    
    criterion_feat = nn.MSELoss()
    
    image_dir = "tianmouc_data/pretrain_images"
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))
        
    train_dataset = TianmoucPretrainDataset(image_paths, T_steps=10, velocity=(4, 2))
    train_data = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    
    for epoch in tqdm(range(epoch_num)):
        backbone.train()
        for step, data in enumerate(train_data):
            cop_seq, td, sd = data
            cop_seq = cop_seq.cuda()
            td = td.cuda()
            sd = sd.cuda()
            
            backbone.reset_stream_state()
            total_loss = 0
            T_steps = td.shape[-1]
            
            for t in range(T_steps):
                is_rgb_available = (t == 0)
                
                current_feat = backbone(
                    rgb_frame=cop_seq[..., 0],
                    sd_slice=sd[..., t], 
                    td_slice=td[..., t], 
                    is_rgb_available=is_rgb_available
                )
                
                current_oracle_rgb = cop_seq[..., t]
                with torch.no_grad():
                    oracle_feat_t = backbone.get_oracle_rgb_feature(current_oracle_rgb)
                
                loss_step = criterion_feat(current_feat, oracle_feat_t)
                total_loss += loss_step
            
            total_loss = total_loss / T_steps
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
