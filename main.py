from tmdat_dataset import TurningDiskDataset
from model import TurningDiskSiamFC
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import time


def train_siamfc():
    epoch_num = 800
    save_period = 2
    load_ckpt_path = ""
    save_ckpt_path = "ckpt/TurningDiskSiamFC_snn.ckpt"
    Path(os.path.dirname(save_ckpt_path)).mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter('summary/train_predictor_{}'.format(int(time.time())))
    net = TurningDiskSiamFC().cuda()
    train_data = DataLoader(TurningDiskDataset(), batch_size=32, pin_memory=True, shuffle=True, num_workers=8)

    if load_ckpt_path:
        net.load_state_dict(torch.load(load_ckpt_path, map_location=torch.device("cuda:0")))

    # net = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3])
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[400], gamma=0.1)

    for epoch in tqdm(range(epoch_num)):
        for step, data in enumerate(train_data):
            cop, td, sd, cop_loc, target_loc = data
            net_out = net(cop.cuda(), sd.cuda(), td.cuda(), cop_loc.cuda(), target_loc.cuda(), training=True)
            optimizer.zero_grad()
            loss = net_out['loss'].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()

            writer.add_scalar('training loss', loss.item(), step + 1 + epoch * len(train_data))

        scheduler.step()
        if (epoch + 1) % save_period == 0:
            print("\rsaving at epoch {}, loss={:.3f}, path: {}".format(epoch + 1, loss.item(), save_ckpt_path))
            torch.save(net.state_dict(), save_ckpt_path)


def test_siamfc():
    load_ckpt_path = ""  # 设为空跳过预训练权重加载，使用随机初始化
    net = TurningDiskSiamFC().cuda()
    if load_ckpt_path:
        net.load_state_dict(torch.load(load_ckpt_path, map_location=torch.device("cuda:0")))
    
    train_data = DataLoader(TurningDiskDataset(test=False), batch_size=1, pin_memory=True, shuffle=False)
    video = cv2.VideoWriter('demo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (640, 320)) 

    for step, data in enumerate(train_data):
        cop, td, sd, cop_loc, target_loc = data
        net_out = net(cop.cuda(), sd.cuda(), td.cuda(), cop_loc.cuda(), target_loc.cuda(), training=False)

        pred_loc = net_out['pred_loc'].squeeze().data.cpu().numpy().astype(np.int64)
        gt_loc = net_out['gt_loc'].squeeze().data.cpu().numpy().astype(np.int64)
        
        # 1. 🚀 核心修复：安全恢复认知底图的像素范围
        cop_np = cop.squeeze().data.cpu().numpy() # [3, 320, 640]
        
        # 如果模型输出或 Dataset 读取出来的数值最大值小于等于 1.0，说明是浮点格式，必须乘以 255 还原！
        if cop_np.max() <= 1.01:
            cop_np = cop_np * 255.0
            
        cop_np = np.clip(cop_np, 0, 255).astype(np.uint8)
        
        # 严格转置并转换为 OpenCV 标准的 BGR 空间（此时白色背景将真正回归为正常的白色）
        base_img = cv2.cvtColor(cop_np.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)

        for t in range(gt_loc.shape[2]):
            plt.gca().clear()
            
            # 克隆底图，防止帧与帧之间互相污染
            img = base_img.copy()
            
            # 3. 绘制目标框（确保坐标在 640x320 画布内部）
            for o in range(gt_loc.shape[0]):
                px, py = pred_loc[o, :, t]
                gx, gy = gt_loc[o, :, t]
                
                px, gx = np.clip([px, gx], 0, 640 - 1)
                py, gy = np.clip([py, gy], 0, 320 - 1)
                
                # 绘制标准的红色预测框 (0, 0, 255) 和绿色真值框 (0, 255, 0)
                img = cv2.rectangle(img, (int(px)-10, int(py)-10), (int(px)+10, int(py)+10), (0, 0, 255), 2)  
                img = cv2.rectangle(img, (int(gx)-10, int(gy)-10), (int(gx)+10, int(gy)+10), (0, 255, 0), 2)  

            # 供 matplotlib 正常预览
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.pause(0.01)
            
            # 写入视频
            video.write(img)

    video.release()

if __name__ == "__main__":
    # train_siamfc()
    test_siamfc()
