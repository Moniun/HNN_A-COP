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
            aop, td, sd, aop_loc, target_loc = data
            net_out = net(aop.cuda(), sd.cuda(), td.cuda(), aop_loc.cuda(), target_loc.cuda(), training=True)
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
    load_ckpt_path = "ckpt/TurningDiskSiamFC_snn.ckpt"
    net = TurningDiskSiamFC().cuda()
    net.load_state_dict(torch.load(load_ckpt_path, map_location=torch.device("cuda:0")))
    train_data = DataLoader(TurningDiskDataset(test=True), batch_size=1, pin_memory=True, shuffle=False)
    video = cv2.VideoWriter('demo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 15, (320, 640))  # 天眸c原生分辨率

    for step, data in enumerate(train_data):
        aop, td, sd, aop_loc, target_loc = data
        net_out = net(aop.cuda(), sd.cuda(), td.cuda(), aop_loc.cuda(), target_loc.cuda(), training=False)

        pred_loc = net_out['pred_loc'].squeeze().data.cpu().numpy().astype(np.int64)
        gt_loc = net_out['gt_loc'].squeeze().data.cpu().numpy().astype(np.int64)
        aop = aop.squeeze().data.cpu().numpy().astype(np.uint8)
        
        # 可视化SD事件（取正极性通道）
        sd_vis = sd.squeeze()[0].data.cpu().numpy().astype(np.uint8)

        for t in range(gt_loc.shape[2]):
            plt.gca().clear()
            img = cv2.cvtColor(aop.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)  # 转换为BGR用于OpenCV
            # 在图像上叠加AOP帧和SD事件
            sd_resized = cv2.resize(sd_vis[..., t], (640, 320))
            img[sd_resized != 0, 2] = 255  # 红色标记事件
            
            for o in range(gt_loc.shape[0]):
                px, py = pred_loc[o, :, t]
                gx, gy = gt_loc[o, :, t]
                img = cv2.rectangle(img, (px-10, py-10), (px+10, py+10), (0, 0, 255), 1)  # 红色预测框
                img = cv2.rectangle(img, (gx-10, gy-10), (gx+10, gy+10), (0, 255, 0), 1)  # 绿色真实框

            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.pause(0.1)
            video.write(img)
    video.release()


if __name__ == "__main__":
    # train_siamfc()
    test_siamfc()
