from tmdat_dataset import TianmoucStreamingDataset
from models import TianmoucHNNBackbone, DetectionHead, FullModel
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import time


def train():
    epoch_num = 10
    save_period = 1
    load_ckpt_path = ""
    save_ckpt_path = "ckpt/HNN_detection_head.ckpt"
    Path(os.path.dirname(save_ckpt_path)).mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter('summary/train_predictor_{}'.format(int(time.time())))
    
    backbone = TianmoucHNNBackbone().cuda()
    task_head = DetectionHead(in_channels=512, num_objects=3).cuda()
    
    if load_ckpt_path:
        state_dict = torch.load(load_ckpt_path, map_location=torch.device("cuda:0"))
        backbone.load_state_dict(state_dict.get('backbone', {}), strict=False)
        task_head.load_state_dict(state_dict.get('head', {}), strict=False)
    
    optimizer = torch.optim.Adam(list(backbone.parameters()) + list(task_head.parameters()), lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[400], gamma=0.1)
    criterion = nn.MSELoss()
    
    train_data = DataLoader(TianmoucStreamingDataset(test=False), batch_size=32, shuffle=True, num_workers=8)
    
    for epoch in tqdm(range(epoch_num)):
        for step, data in enumerate(train_data):
            cop, td, sd, cop_loc, target_loc = data
            cop = cop.cuda()
            td = td.cuda()
            sd = sd.cuda()
            target_loc = target_loc.cuda().float()
            
            backbone.reset_stream_state()
            total_loss = 0
            
            T_steps = td.shape[-1]
            for t in range(T_steps):
                is_rgb_available = (t == 0)
                
                current_feat = backbone(
                    rgb_frame=cop, 
                    sd_slice=sd[..., t], 
                    td_slice=td[..., t], 
                    is_rgb_available=is_rgb_available
                )
                
                pred_loc = task_head(current_feat)
                loss_step = criterion(pred_loc, target_loc[..., t])
                total_loss += loss_step
            
            total_loss = total_loss / T_steps
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(task_head.parameters()), 1)
            optimizer.step()

            writer.add_scalar('training loss', total_loss.item(), step + 1 + epoch * len(train_data))

        scheduler.step()
        if (epoch + 1) % save_period == 0:
            print("\rsaving at epoch {}, loss={:.3f}, path: {}".format(epoch + 1, total_loss.item(), save_ckpt_path))
            torch.save({
                'backbone': backbone.state_dict(),
                'head': task_head.state_dict()
            }, save_ckpt_path)


def test():
    load_ckpt_path = ""
    backbone = TianmoucHNNBackbone().cuda()
    task_head = DetectionHead(in_channels=512, num_objects=3).cuda()
    
    if load_ckpt_path:
        state_dict = torch.load(load_ckpt_path, map_location=torch.device("cuda:0"))
        backbone.load_state_dict(state_dict.get('backbone', {}), strict=False)
        task_head.load_state_dict(state_dict.get('head', {}), strict=False)
    
    test_data = DataLoader(TianmoucStreamingDataset(test=True), batch_size=1, pin_memory=True, shuffle=False)
    video = cv2.VideoWriter('demo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (640, 320)) 

    for step, data in enumerate(test_data):
        cop, td, sd, cop_loc, target_loc = data
        cop = cop.cuda()
        td = td.cuda()
        sd = sd.cuda()
        
        backbone.reset_stream_state()
        T_steps = td.shape[-1]
        
        pred_loc_list = []
        for t in range(T_steps):
            is_rgb_available = (t == 0)
            current_feat = backbone(
                rgb_frame=cop, 
                sd_slice=sd[..., t], 
                td_slice=td[..., t], 
                is_rgb_available=is_rgb_available
            )
            pred_loc = task_head(current_feat)
            pred_loc_list.append(pred_loc.squeeze().data.cpu().numpy())
        
        pred_loc_arr = np.stack(pred_loc_list, axis=-1).astype(np.int64)
        gt_loc = target_loc.squeeze().numpy().astype(np.int64)
        
        cop_np = cop.squeeze().data.cpu().numpy()
        if cop_np.max() <= 1.01:
            cop_np = cop_np * 255.0
        cop_np = np.clip(cop_np, 0, 255).astype(np.uint8)
        base_img = cv2.cvtColor(cop_np.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)

        for t in range(T_steps):
            plt.gca().clear()
            img = base_img.copy()
            
            for o in range(3):
                px, py = pred_loc_arr[o, :, t]
                gx, gy = gt_loc[o, :, t]
                
                px, gx = np.clip([px, gx], 0, 640 - 1)
                py, gy = np.clip([py, gy], 0, 320 - 1)
                
                img = cv2.rectangle(img, (int(px)-10, int(py)-10), (int(px)+10, int(py)+10), (0, 0, 255), 2)  
                img = cv2.rectangle(img, (int(gx)-10, int(gy)-10), (int(gx)+10, int(gy)+10), (0, 255, 0), 2)  

            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.pause(0.01)
            video.write(img)

    video.release()


if __name__ == "__main__":
    train()
    # test()