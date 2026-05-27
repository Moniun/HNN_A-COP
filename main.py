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
    video = cv2.VideoWriter('demo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 15, (640, 320)) # 天眸c原生分辨率

    for step, data in enumerate(train_data):
        cop, td, sd, cop_loc, target_loc = data
        net_out = net(cop.cuda(), sd.cuda(), td.cuda(), cop_loc.cuda(), target_loc.cuda(), training=False)

        pred_loc = net_out['pred_loc'].squeeze().data.cpu().numpy().astype(np.int64)
        gt_loc = net_out['gt_loc'].squeeze().data.cpu().numpy().astype(np.int64)
        cop = cop.squeeze().data.cpu().numpy().astype(np.uint8)
        
        # 可视化SD事件（取正极性通道）
        # sd_vis = sd.squeeze()[0].data.cpu().numpy().astype(np.uint8)

        # ---------- 替换原来的 sd_vis 和画图逻辑 ----------
        
        # 💡 修复点 1：提取SD信号时先取绝对值，并用 >0.5 过滤掉底层暗噪声，防止负数变成 255
        sd_np = sd.squeeze()[0].data.cpu().numpy()
        sd_vis = (np.abs(sd_np) > 0.5).astype(np.uint8)

        for t in range(gt_loc.shape[2]):
            plt.gca().clear()
            img = cop.transpose(1, 2, 0).copy() 
            
            # 💡 修复点 2：事件流插值必须使用 INTER_NEAREST（最近邻），否则脉冲会被糊成一团
            sd_resized = cv2.resize(sd_vis[..., t], (640, 320), interpolation=cv2.INTER_NEAREST)
            
            # 此时再用大于 0 作为掩码去上红色，就只会点亮真正的运动边缘了
            img[sd_resized > 0, 2] = 255  
            
            for o in range(gt_loc.shape[0]):
                px, py = pred_loc[o, :, t]
                gx, gy = gt_loc[o, :, t]
                img = cv2.rectangle(img, (px-10, py-10), (px+10, py+10), (0, 0, 255), 1)  # 红色预测框
                img = cv2.rectangle(img, (gx-10, gy-10), (gx+10, gy+10), (0, 255, 0), 1)  # 绿色真实框

            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.pause(0.1)
            video.write(img)
        # ------------------------------------------------
    video.release()


if __name__ == "__main__":
    # train_siamfc()
    test_siamfc()
